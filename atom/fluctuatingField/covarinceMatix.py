import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import simpson
from atom import utils
import multiprocessing
import warnings
import pandas as pd


class CovarianceMatrices:
    """
    This class is responsible for calculating the covariance matrices used in spatial covariance function calculations.

    Attributes:
        modelGrid (xr.Dataset): Contains the model grid data.
        atarray (xr.Dataset): Contains acoustic tomography array data.
        linearSystem (xr.Dataset): Contains the bulk linear system data.
        sigmaT (float): Standard deviation of temperature. Default is 0.14.
        lT (float): Length scale of temperature. Default is 15.0.
        sigmaU (float): Standard deviation of u-component of wind. Default is 0.72.
        sigmaV (float): Standard deviation of v-component of wind. Default is 0.42.
        l (float): Length scale of wind. Default is 15.0.
        nFrames (int): Number of *additional* frames to consider in covariance matrices and TDSI (the N in N+1 frames). Default is 0.
        alignment (str): Alignment of the stencil selecting frames. Default is 'center'.
    """

    def __init__(
        self,
        configDict: dict,
        modelGrid: xr.Dataset,
        atarray: xr.Dataset,
        linearSystem: xr.Dataset,
    ):
        for key, value in configDict.items():
            setattr(self, key, value)

        self.modelGrid = modelGrid
        self.atarray = atarray

        # Initialize the xarray dataset
        self.ds = xr.Dataset(
            data_vars={
                "cBulk": linearSystem.c,
                "TBulk": linearSystem.T,
                "advectionVelocity": linearSystem[["u", "v"]].to_array(),
            }
        )

        self.ds.attrs = configDict

        if not self._checkDomainLimits():
            raise ValueError(
                "Model grid does not cover the domain of the path data. Please adjust the model grid."
            )

        # Define stencil used for advection scheme
        self.stencil, self.frameSets = _defineStencil(
            nFrames=self.ds.nFrames,
            alignment=self.ds.alignment,
            totalFrames=len(linearSystem.frame),
            advectionScheme=self.ds.advectionScheme,
        )

    def covarianceFunction(
        self,
        modelGrid: xr.Dataset = None,
        sigma: float = None,
        l: float = None,
        component: str = None,
    ):
        """
        Calculates the spatial covariance function given a model grid and specific parameters. When timeDelay ~= 0, the second spatial coordinate in the covariance function r' is shifted by the advection velocity and time delay  r' + U delta t, assuming Taylor's frozen field hypothesis.

        Parameters:
            modelGrid (xr.Dataset, optional): The dataset containing the model grid with variables "x" and "y".
                Default is None.
            sigma (float, optional): Standard deviation of the process. For "uv" correlation, a length 2 array
                representing standard deviation for x and y is expected. Default is None.
            l (float, optional): Length scale of the process. Default is None.
            component (str, optional): Specifies the type of component to be used for covariance calculation.
                Valid inputs are "TT", "uu", "vv", and "uv". Default is None.
            timeDelay (float, optional): time delay between frames in seconds. Default is 0.0.
            advectionVelocity (np.ndarray, optional): advection velocity

        Returns:
            numpy.ndarray: The calculated covariance matrix for the specified component.

        Raises:
            AssertionError: If the component is "uv" and sigma is not a length 2 array.

        Notes:
            If the component is None, the function will prompt the user to specify a component.
        """
        x = modelGrid.modelGrid.sel(variable="x").to_numpy().flatten()
        y = modelGrid.modelGrid.sel(variable="y").to_numpy().flatten()
        dx = (x - (x[:, None])).reshape([len(np.unique(x)), len(np.unique(y)), -1])
        dy = (y - (y[:, None])).reshape([len(np.unique(x)), len(np.unique(y)), -1])

        # Calculate the Euclidean distance between all pairs of points
        dr = np.sqrt(dx**2 + dy**2)

        def covar(dr, sigma, l):
            # Gaussian function representing normally distributed covariances
            # standard deviation should account for multiple components for 'uv'
            if np.isscalar(sigma):
                sigSq = sigma**2
            elif len(sigma) == 2:
                sigSq = np.prod(sigma)

            # return normal distribution scaled by sigma**2 and l**2
            return sigSq * np.exp(-(dr**2) / l**2)

        if component is None:
            ValueError("please specify a component (T, u, v, or uv).")
        if component == "TT":
            return covar(dr, sigma, l)
        elif component == "uu":
            return covar(dr, sigma, l) * (1 - dy**2 / l**2)
        elif component == "vv":
            return covar(dr, sigma, l) * (1 - dx**2 / l**2)
        elif component == "uv":
            assert (
                len(sigma) == 2
            ), "uv correlelation expects standard deviation for x and y, pls supply a length 2 array."
            return covar(dr, sigma, l) * (dx * dy) / l**2

    def calculateCovariances(self) -> None:
        """
        Calculates and assigns the covariance function values to the current dataset object.

        Notes:
            This method does not return any value, but directly modifies the dataset within the object instance.
            It expects that the following attributes are pre-defined in the dataset: modelGrid, sigmaT, lT, sigmaU,
            sigmaV, and l. The 'modelXY' coordinate is created by stacking 'x' and 'y' from the dataset.
        """

        x, y = self.modelGrid.unstack().x, self.modelGrid.unstack().y
        modelXY = self.modelGrid.modelXY

        # Covariance function values
        self.ds["B_TT"] = xr.DataArray(
            data=self.covarianceFunction(
                modelGrid=self.modelGrid,
                sigma=self.ds.sigmaT,
                l=self.ds.lT,
                component="TT",
            ),
            coords={
                "x": x,
                "y": y,
                "modelXY": modelXY.values,
            },
        )
        self.ds["B_uu"] = xr.DataArray(
            data=self.covarianceFunction(
                modelGrid=self.modelGrid,
                sigma=self.ds.sigmaU,
                l=self.ds.l,
                component="uu",
            ),
            coords={
                "x": x,
                "y": y,
                "modelXY": modelXY.values,
            },
        )
        self.ds["B_vv"] = xr.DataArray(
            data=self.covarianceFunction(
                modelGrid=self.modelGrid,
                sigma=self.ds.sigmaV,
                l=self.ds.l,
                component="vv",
            ),
            coords={
                "x": x,
                "y": y,
                "modelXY": modelXY.values,
            },
        )
        self.ds["B_uv"] = xr.DataArray(
            data=self.covarianceFunction(
                modelGrid=self.modelGrid,
                sigma=np.array([self.ds.sigmaU, self.ds.sigmaV]),
                l=self.ds.l,
                component="uv",
            ),
            coords={
                "x": x,
                "y": y,
                "modelXY": modelXY.values,
            },
        )

    def modelDataCovarianceMatrix(self) -> None:
        """
        Calculates the model-data covariance matrix and adds it to the instance's dataset.

        Notes:
            This method does not return any value, but modifies the 'ds' attribute of the class instance by
            adding the 'Bmd' data array.
            It expects that the following attributes are pre-defined in the instance: 'pathData', 'ds',
            'modelGrid', 'nModelPointsX', 'nModelPointsY', 'atarray.ds.pathOrientation'.
            The 'ds' attribute is an xarray Dataset instance and 'atarray' is an attribute of the class where
            this method is defined.
        """
        ## reshape interpolated point array
        nTdsiFrames, nPaths, nPoints, nComponents = self.ds.interpolationPoints.shape
        J = self.modelGrid.nx * self.modelGrid.ny

        points = self.ds.interpolationPoints.to_numpy().reshape([-1, nComponents])
        x, y = np.unique(self.modelGrid.x.values), np.unique(self.modelGrid.y.values)

        TBulk = self.ds.TBulk.mean(dim="frame").values
        cBulk = self.ds.cBulk.mean(dim="frame").values

        # setup intermediate dataset for storage
        dsBmd = xr.Dataset()

        coords = dict(
            tdsiFrame=self.ds.tdsiFrame,
            pathID=self.atarray.pathID,
            modelXY=self.modelGrid.modelXY,
        )

        # Compute path integrals over covariance matrices
        for component in ["B_TT", "B_uu", "B_vv", "B_uv"]:
            covar = self.ds[component].to_numpy()

            ## Covariance interpolated along paths
            interp = RegularGridInterpolator((x, y), covar)

            ## interpolate covariance
            pathVals = interp(points)
            pathVals = pathVals.reshape([nTdsiFrames, nPaths, nPoints, J])

            ## integrate covariance function centered at model grid location, along each path
            if component == "B_TT":
                dsBmd[f"{component}_md"] = xr.DataArray(
                    data=(
                        cBulk
                        / (2 * TBulk)
                        * simpson(
                            pathVals,
                            dx=self.atarray.integraldx.to_numpy()[None, :, None],
                            axis=2,
                        )
                    ),
                    coords=coords,
                )
            else:
                dsBmd[f"{component}_md"] = xr.DataArray(
                    data=simpson(
                        pathVals,
                        dx=self.atarray.integraldx.to_numpy()[None, :, None],
                        axis=2,
                    ),
                    coords=coords,
                )

        # # Combine path integrals across covariance matrices to form Model-data covariance matrix Bmd
        # dsBmd = dsBmd.stack(modelXY=["x", "y"])

        dsBmd["B_EWComponent"] = (
            (
                dsBmd["B_uu_md"].unstack(dim="pathID")
                * np.cos(self.atarray.pathOrientation.unstack())
                + dsBmd["B_uv_md"].unstack(dim="pathID")
                * np.sin(self.atarray.pathOrientation.unstack())
            )
            .stack(pathID=["spk", "mic"])
            .dropna(dim="pathID")
        )

        dsBmd["B_NSComponent"] = (
            (
                dsBmd["B_uv_md"].unstack(dim="pathID")
                * np.cos(self.atarray.pathOrientation.unstack())
                + dsBmd["B_vv_md"].unstack(dim="pathID")
                * np.sin(self.atarray.pathOrientation.unstack())
            )
            .stack(pathID=["spk", "mic"])
            .dropna(dim="pathID")
        )

        Bmd = xr.concat(
            [dsBmd["B_TT_md"], dsBmd["B_EWComponent"], dsBmd["B_NSComponent"]],
            pd.Index(["T", "u", "v"], name="component"),
        )

        return Bmd

    def dataDataCovarianceMatrix(self) -> xr.DataArray:
        """
        Calculates the data-data covariance matrix and adds it to the instance's dataset.

        Notes:
            This method does not return any value, but modifies the 'ds' attribute of the class instance by
            adding the 'Bdd' data array.
            It expects that the following attributes are pre-defined in the instance: 'pathData', 'ds',
            'modelGrid', 'nModelPointsX', 'nModelPointsY', 'ds.pathOrientation'.
            The 'ds' attribute is an xarray Dataset instance and 'modelGrid' is an attribute of the class where
            this method is defined.
        """

        ## reshape interpolated point array
        nTdsiFrames, nPaths, nPoints, nComponents = self.ds.interpolationPoints.shape
        J = self.modelGrid.nx * self.modelGrid.ny

        points = self.ds.interpolationPoints.to_numpy().reshape([-1, nComponents])
        x, y = np.unique(self.modelGrid.x.values), np.unique(self.modelGrid.y.values)

        TBulk = self.ds.TBulk.mean(dim="frame").values
        cBulk = self.ds.cBulk.mean(dim="frame").values

        dummyCoords = self.makeDummyCoords()

        # setup intermediate dataset for storage
        dsBdd = xr.Dataset()

        po = self.atarray.pathOrientation.to_numpy()

        for component in ["B_TT", "B_uu", "B_vv", "B_uv", "B_vu"]:
            # Compute path integrals over covariance matrices
            if component == "B_vu":
                covar = self.ds["B_uv"].to_numpy()
            else:
                covar = self.ds[component].to_numpy()

            ## Interpolate along paths
            interp = RegularGridInterpolator((x, y), covar)
            pathVals = interp(points)
            pathVals = pathVals.reshape([nTdsiFrames, nPaths, nPoints, J])

            # perform first integration
            if component == "B_TT":
                intermed = (
                    simpson(
                        pathVals,
                        dx=self.atarray.integraldx.to_numpy()[None, :, None],
                        axis=2,
                    )
                    .reshape(
                        [nTdsiFrames, nPaths, self.modelGrid.nx, self.modelGrid.ny]
                    )
                    .transpose([2, 3, 1, 0])
                )
            elif component == "B_uu" or component == "B_uv":
                intermed = (
                    simpson(
                        pathVals * np.cos(po[None, :, None, None]),
                        dx=self.atarray.integraldx.to_numpy()[None, :, None],
                        axis=2,
                    )
                    .reshape(
                        [nTdsiFrames, nPaths, self.modelGrid.nx, self.modelGrid.ny]
                    )
                    .transpose([2, 3, 1, 0])
                )
            else:
                intermed = (
                    simpson(
                        pathVals * np.sin(po[None, :, None, None]),
                        dx=self.atarray.integraldx.to_numpy()[None, :, None],
                        axis=2,
                    )
                    .reshape(
                        [nTdsiFrames, nPaths, self.modelGrid.nx, self.modelGrid.ny]
                    )
                    .transpose([2, 3, 1, 0])
                )

            ## re-interpolate along paths
            interp = RegularGridInterpolator((np.unique(x), np.unique(y)), intermed)
            pathVals = interp(points)
            pathVals = pathVals.reshape(
                [nTdsiFrames, nPoints, nPaths, nPaths, nTdsiFrames]
            )

            # perform second integration
            if component == "B_TT":
                intermed = simpson(
                    pathVals,
                    dx=self.atarray.integraldx.to_numpy()[None, :, None, None],
                    axis=1,
                )
            elif component == "B_uu" or component == "B_vu":
                intermed = simpson(
                    pathVals * np.cos(po[:, None, None]),
                    dx=self.atarray.integraldx.to_numpy()[:, None],
                    axis=1,
                )
            else:
                intermed = simpson(
                    pathVals * np.sin(po[:, None, None]),
                    dx=self.atarray.integraldx.to_numpy()[:, None],
                    axis=1,
                )

            dsBdd[f"{component}_dd"] = xr.DataArray(
                data=intermed,
                coords={
                    "tdsiFrame": self.ds.tdsiFrame,
                    "pathID": self.atarray.pathID,
                    "pathID_duplicate": dummyCoords["pathID_duplicate"],
                    "tdsiFrame_duplicate": dummyCoords["tdsiFrame_duplicate"],
                },
            )

        Bdd = (
            cBulk**2 / (4 * TBulk**2) * dsBdd["B_TT_dd"]
            + dsBdd["B_uu_dd"]
            + dsBdd["B_vv_dd"]
            + dsBdd["B_uv_dd"]
            + dsBdd["B_vu_dd"]
        )

        return Bdd

    def assembleTDSICovarianceMatrices(self):
        """
        Assembles the TDSI (Time-Dependent Stochastic Inversion) covariance matrices.

        Args:
            nFrames (int, optional): The number of frames to consider. If not provided, it uses the value of self.nFrames.
            alignment (str, optional): The alignment preference for time delays. Accepts 'center', 'forward', or 'backward'.
                If not provided, it uses the value of self.alignment.

        Raises:
            ValueError: If the alignment type is not recognized.

        Returns:
            None
        """

        if self.covarianceSource == "model":
            # calculate B_TT, B_uu, B_vv, B_uv
            self.calculateCovariances()
        elif self.covarianceSource == "data":
            self.loadCovariances(self.covarianceFilePath)

        interpolationPoints = []

        if self.ds.advectionScheme == "stationary":
            advectionVelocity = self.ds.advectionVelocity.mean(dim="frame").values
            timeDeltas = self.ds.timeDelay * self.frameSets

            advectionDistance = advectionVelocity[:, None] * timeDeltas[None, :]
            advectionDistance = xr.DataArray(
                data=advectionDistance,
                coords={"coord": ["easting", "northing"], "tdsiFrame": self.frameSets},
            )

            for offset in advectionDistance.transpose("tdsiFrame", "coord"):
                interpolationPoints.append(
                    self.atarray.integralPoints.sel(coord=["easting", "northing"])
                    + offset
                )
            self.ds["interpolationPoints"] = xr.concat(
                interpolationPoints, dim="tdsiFrame"
            )

            # calculate Bmd
            self.ds["Bmd"] = self.modelDataCovarianceMatrix()

            # calculate Bdd
            Bdd = self.dataDataCovarianceMatrix()
            self.ds["Bdd"] = Bdd

        # #TODO generalizie the method to non-stationary conditions ...
        # elif self.ds.advectionScheme == "general":
        #     pass

    def loadCovariances(self, filePath: str) -> None:
        """
        Loads covariance data from a specified file and stores it in the instance's dataset.

        Args:
            filePath (str): The path to the file containing the covariance data.

        Returns:
            None
        """
        # Load the covariance data from the file
        corr = xr.load_dataset(filePath)

        renameVars = {"u": "uu", "v": "vv", "w": "ww", "T": "TT"}
        for var in renameVars:
            if var in list(corr.data_vars):
                corr = corr.rename({var: renameVars[var]})

        # Store the loaded data in the instance's dataset
        self.ds["B_TT"] = corr["TT"]
        self.ds["B_uu"] = corr["uu"]
        self.ds["B_vv"] = corr["vv"]
        self.ds["B_uv"] = corr["uv"]

    def _checkDomainLimits(
        self,
    ):
        """
        Checks if the domain limits are within the domain of the data.

        Args:
        modelx (numpy.ndarray): x-coordinates of the data.
        modely (numpy.ndarray): y-coordinates of the data.
        xlim (list): x-limits of the domain.
        ylim (list): y-limits of the domain.

        Returns:
        bool: True if the domain limits are within the domain of the data.
        """

        pathXY = self.atarray.integralPoints[..., :-1].to_numpy().reshape([-1, 2])
        minx, miny = pathXY.min(axis=0)
        maxx, maxy = pathXY.max(axis=0)

        advDistance = (
            self.ds.advectionVelocity.mean(dim="frame")
            * self.ds.timeDelay
            * self.ds.nFrames
        )

        xlim = np.array([minx, maxx]) + np.array([-advDistance[0], advDistance[0]])
        ylim = np.array([miny, maxy]) + np.array([-advDistance[1], advDistance[1]])

        modelx, modely = self.modelGrid.x, self.modelGrid.y

        if (modelx.min() > xlim[0]) | (modelx.max() < xlim[1]):
            print(
                f"Model grid does not cover the domain of the advected data. Limits should be {xlim} and {ylim}."
            )
            return False
        elif (modely.min() > ylim[0]) | (modely.max() < ylim[1]):
            print(
                f"Model grid does not cover the domain of the advected data. Limits should be {xlim} and {ylim}."
            )
            return False
        else:
            return True

    def makeDummyCoords(self):
        """
        make dummy coordinates for the model data so that it can be stacked, unstacked, and subselected without errors.
        """
        tsf = self.frameSets
        spk = self.atarray.unstack().spk.values
        mic = self.atarray.unstack().mic.values

        tmp = xr.DataArray(
            data=np.zeros([len(tsf), len(spk), len(mic)]),
            coords={
                "tdsiFrame_duplicate": tsf,
                "spk_duplicate": spk,
                "mic_duplicate": mic,
            },
        )
        tmp = (
            tmp.where(tmp.spk_duplicate != tmp.mic_duplicate)
            .stack(
                pathID_duplicate=[
                    "spk_duplicate",
                    "mic_duplicate",
                ]
            )
            .dropna("pathID_duplicate", how="all")
        )
        return tmp.coords

    ## utility functions
    def to_netcdf(self, filePath) -> None:
        # need to unstack all indices
        utils.to_netcdf(self.ds, filePath)

    @classmethod
    def from_netcdf(self, filePath) -> None:
        # need to restack all indices
        self.ds = utils.from_netcdf(self.ds, filePath)

    def to_pickle(self, file_path):
        utils.save_to_pickle(self, file_path)

    @classmethod
    def from_pickle(cls, file_path):
        obj = utils.load_from_pickle(file_path)
        return obj

    def describe(self):
        return utils.describe_dataset(self.ds)


def _defineStencil(
    nFrames: int = 0,
    totalFrames: int = 10,
    alignment: str = "center",
    advectionScheme: str = "stationary",
):
    """
    Defines a stencil array for selecting frames from a sequence.

    The stencil array contains indices for selecting frames centered around
    a given frame, with configurable number of frames and alignment. Used
    to construct covariance matrices from frame sequences.

    """

    if (alignment == "center") and (nFrames % 2 != 0):
        warnings.warn(
            f"For alignment=='center' and an odd number of nFrames, frame sets are being offset by 1 to the future, e.g., nframes=3 -> [-1,0,1,2]"
        )
        nf = nFrames + 2
    else:
        nf = nFrames + 1

    if alignment == "center":
        stencil = (np.arange(nf) - np.floor(nf / 2)).astype(int)
        if nFrames % 2 != 0:
            stencil = stencil[1:]
    elif alignment == "forward":
        stencil = np.arange(nFrames).astype(int)
    elif (alignment == "backward") | (alignment == "reverse"):
        stencil = (np.arange(nFrames) - nFrames + 1).astype(int)
    else:
        ValueError(f"Alignment type {alignment} not recognized.")

    if advectionScheme == "stationary":
        frameSets = np.arange(-(len(stencil) - 1), len(stencil))
    elif advectionScheme == "general":
        frameSets = np.zeros((totalFrames, nFrames))
        for ii in range(totalFrames):
            frameSets[ii, :] = ii + stencil
            if frameSets[ii, :].min() < 0:
                frameSets[ii, :] -= frameSets[ii, :].min()
            if frameSets[ii, :].max() > totalFrames - 1:
                frameSets[ii, :] -= frameSets[ii, :].max() - totalFrames + 1
    frameSets = frameSets.astype(int)

    return stencil, frameSets


def _distributeProcesses(function, arguments):
    """placeholder for now
    #TODO someday..."""

    # Use two less than the total number of CPU cores
    num_processes = multiprocessing.cpu_count() - 2

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Distribute the function calls across processes using starmap
        results = pool.starmap(function, arguments)

    return results
