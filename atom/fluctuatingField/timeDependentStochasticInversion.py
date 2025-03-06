import numpy as np
import xarray as xr
from atom import utils
from typing import Iterable
from atom import utils


class TimeDependentStochasticInversion:
    """
    Instantiates the TimeDependentStochasticInversion class with the provided datasets.

    Parameters:
        modelGrid (xr.Dataset): The dataset containing the model grid.
        atarray (xr.Dataset): The dataset containing the path data.
        covarMatrices (xr.Dataset): The dataset containing the covariance matrices.
    """

    def __init__(
        self,
        modelGrid: xr.Dataset,
        atarray: xr.Dataset,
        covarMatrices: xr.Dataset,
        bulkFlowData: xr.Dataset,
        stencil: Iterable[int] = [-1, 0, 1],
        frameSets: Iterable[int] = [-2, -1, 0, 1, 2],
    ):
        self.modelGrid = modelGrid
        self.atarray = atarray
        self.covarMatrices = covarMatrices
        self.bulkFlowData = bulkFlowData
        self.stencil = stencil
        self.frameSets = frameSets
        self.ds = xr.Dataset()

    def optimalStochasticInverseOperator(self):
        """
        Calculates the optimal stochastic inverse operator and adds it to the instance's dataset.

        This method computes the matrix product of the reshaped model-data covariance matrix and the inverse of the data-data covariance matrix. The resulting matrix 'A', representing the optimal stochastic inverse operator, is then reshaped and stored in the instance's dataset as 'A'.

        Notes:
            This method does not return any value, but modifies the 'ds' attribute of the class instance by adding the 'A' data array.
            It expects that the following attributes are pre-defined in the instance: 'ds', 'ds.Rmd', 'ds.Rdd', 'nModelPointsX', 'nModelPointsY', and 'atarray'.
            The 'ds' attribute is an xarray Dataset instance.
        """

        Rdd = self.covarMatrices.Bdd.sel(
            tdsiFrame=self.stencil, tdsiFrame_duplicate=self.stencil
        ).transpose("tdsiFrame", "pathID", "tdsiFrame_duplicate", "pathID_duplicate")

        Rdd = (
            Rdd.unstack()
            .stack(
                pathID=["tdsiFrame", "spk", "mic"],
                pathID_duplicate=[
                    "tdsiFrame_duplicate",
                    "spk_duplicate",
                    "mic_duplicate",
                ],
            )
            .dropna(dim="pathID", how="all")
            .dropna(dim="pathID_duplicate", how="all")
        )

        # Calculate the Moore-Penrose pseudo-inverse of Rdd
        RddInv = xr.DataArray(
            data=np.linalg.pinv(Rdd),
            coords=Rdd.coords,
        )

        Rmd = self.covarMatrices.Bmd.sel(tdsiFrame=self.stencil)
        Rmd = (
            Rmd.unstack()
            .stack(
                pathID=["tdsiFrame", "spk", "mic"], modelVar=["component", "modelXY"]
            )
            .dropna(dim="pathID", how="all")
            .T
        )

        A = xr.DataArray(
            data=np.dot(Rmd, RddInv),
            coords=dict(
                modelVar=Rmd.modelVar,
                pathID=Rmd.pathID,
            ),
        ).dropna(dim="pathID", how="all")

        self.ds = A.to_dataset(
            name="stochasticInverseOperator",
        )

    def assembleDataVector(self):
        """
        Assembles the data vector `d`.

        Notes:
            This method does not return any value, but modifies the 'ds' attribute of the class instance by adding the 'd' data array.
            The data vector `d` is assembled as:

            .. math::

                \mathbf{d} = \left[ \mathbf{d}(t - N \tau)~;~ \mathbf{d}(t - N \tau+\tau)~;~\hdots ~;~ \mathbf{d}(t) \right]

        """
        bulkFlowTerm = self.atarray.pathLength * (
            self.bulkFlowData.c
            - self.bulkFlowData.u * np.cos(self.atarray.pathOrientation)
            - self.bulkFlowData.v * np.sin(self.atarray.pathOrientation)
        )

        speedOfSoundTerm = self.bulkFlowData.c**2 * self.bulkFlowData.measuredTravelTime

        dataVector = (
            (bulkFlowTerm.unstack() - speedOfSoundTerm.unstack())
            .stack(pathID=["spk", "mic"])
            .dropna(dim="pathID")
        )

        if self.covarMatrices.nFrames > 0:
            # create array of time delays of length nFrames
            frameSets = self.getFrameSets()
            # tdsiFrames = np.arange(len(self.stencil)).astype(int)
            pathID = self.ds.pathID

            dataVectorList = [
                xr.DataArray(
                    data=dataVector.isel(frame=n)
                    .unstack()
                    .rename({"frame": "tdsiFrame"})
                    .stack(pathID=["tdsiFrame", "spk", "mic"])
                    .dropna(dim="pathID", how="all")
                    .values,
                    coords={"pathID": pathID},
                )
                for n in frameSets
            ]
            dataVector = xr.concat(dataVectorList, dim="frame")

        dataVector["frame"] = self.covarMatrices.frame
        self.ds["dataVector"] = dataVector

    def repackDataVector(self):

        dataVector = self.ds.dataVector
        frames = np.arange(len(self.ds.frame)).astype(int)
        frameSets = self.getFrameSets()

        dataBlock = np.zeros()

    def getFrameSets(self):
        nFrames = len(self.stencil)
        totalFrames = len(self.covarMatrices.frame)
        stencil = np.array(self.stencil).astype(int)
        frameSets = np.zeros((totalFrames, nFrames))
        for ii in range(totalFrames):
            frameSets[ii, :] = ii + stencil
            if frameSets[ii, :].min() < 0:
                frameSets[ii, :] -= frameSets[ii, :].min()
            if frameSets[ii, :].max() > totalFrames - 1:
                frameSets[ii, :] -= frameSets[ii, :].max() - totalFrames + 1
        frameSets = frameSets.astype(int)
        return frameSets

    def calculateFluctuatingFields(self):
        """
        Calculates the fluctuating fields and adds it to the instance's dataset.

        Notes:
            This method does not return any value, but modifies the 'ds' attribute of the class instance by adding the 'm' data array.
            The fluctuating fields `m` are calculated as:

            .. math::

                m = \mathbf{A} \mathbf{d}

            where `m` contains all of the fluctuating field data :math:`m = [T'(r,t),u'(r,t),v'(r,t)]`.
        """

        self.ds["modelValues"] = self.ds.stochasticInverseOperator.dot(
            self.ds.dataVector
        )

        # self.ds["T"] = self.ds["modelValues"].sel(variable="T")
        # self.ds["u"] = self.ds["modelValues"].sel(variable="u")
        # self.ds["v"] = self.ds["modelValues"].sel(variable="v")

        # ds = self.ds.drop("modelXY")

        # ds.coords["x"] = self.modelGrid.unstack()["x"]
        # ds.coords["y"] = self.modelGrid.unstack()["y"]

        # self.ds = ds.stack(modelXY=["x", "y"])

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
