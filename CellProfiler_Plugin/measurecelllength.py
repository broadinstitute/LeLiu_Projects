import centrosome.cpmorphology
import centrosome.zernike
import numpy
import scipy.ndimage
import skimage.measure
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT
from cellprofiler_core.module import Module
from cellprofiler_core.object import Objects
from cellprofiler_core.setting import Divider, Binary, ValidationError
from cellprofiler_core.setting.subscriber import LabelListSubscriber
import cellprofiler.gui.help.content
import cellprofiler.icons


"""The category of the per-object measurements made by this module"""
AREA_SHAPE = "AreaShape"

"""Calculate Zernike features for N,M where N=0 through ZERNIKE_N"""
ZERNIKE_N = 9

F_AREA = "Area"
F_CELL_LEN = "Cell_length"
F_CELL_WIDTH = "Cell_width"
F_PERIMETER = "Perimeter"
F_VOLUME = "Volume"
F_SURFACE_AREA = "SurfaceArea"
F_ECCENTRICITY = "Eccentricity"
F_SOLIDITY = "Solidity"
F_CONVEX_AREA = "ConvexArea"
F_EXTENT = "Extent"
F_CENTER_X = "Center_X"
F_CENTER_Y = "Center_Y"
F_CENTER_Z = "Center_Z"
F_BBOX_AREA = "BoundingBoxArea"
F_BBOX_VOLUME = "BoundingBoxVolume"
F_MIN_X = "BoundingBoxMinimum_X"
F_MAX_X = "BoundingBoxMaximum_X"
F_MIN_Y = "BoundingBoxMinimum_Y"
F_MAX_Y = "BoundingBoxMaximum_Y"
F_MIN_Z = "BoundingBoxMinimum_Z"
F_MAX_Z = "BoundingBoxMaximum_Z"

F_EULER_NUMBER = "EulerNumber"
F_FORM_FACTOR = "FormFactor"
F_MAJOR_AXIS_LENGTH = "MajorAxisLength"
F_MINOR_AXIS_LENGTH = "MinorAxisLength"
F_ORIENTATION = "Orientation"
F_COMPACTNESS = "Compactness"
F_INERTIA = "InertiaTensor"
F_MAXIMUM_RADIUS = "MaximumRadius"
F_MEDIAN_RADIUS = "MedianRadius"
F_MEAN_RADIUS = "MeanRadius"
F_MIN_FERET_DIAMETER = "MinFeretDiameter"
F_MAX_FERET_DIAMETER = "MaxFeretDiameter"

F_CENTRAL_MOMENT_0_0 = "CentralMoment_0_0"
F_CENTRAL_MOMENT_0_1 = "CentralMoment_0_1"
F_CENTRAL_MOMENT_0_2 = "CentralMoment_0_2"
F_CENTRAL_MOMENT_0_3 = "CentralMoment_0_3"
F_CENTRAL_MOMENT_1_0 = "CentralMoment_1_0"
F_CENTRAL_MOMENT_1_1 = "CentralMoment_1_1"
F_CENTRAL_MOMENT_1_2 = "CentralMoment_1_2"
F_CENTRAL_MOMENT_1_3 = "CentralMoment_1_3"
F_CENTRAL_MOMENT_2_0 = "CentralMoment_2_0"
F_CENTRAL_MOMENT_2_1 = "CentralMoment_2_1"
F_CENTRAL_MOMENT_2_2 = "CentralMoment_2_2"
F_CENTRAL_MOMENT_2_3 = "CentralMoment_2_3"
F_EQUIVALENT_DIAMETER = "EquivalentDiameter"
F_HU_MOMENT_0 = "HuMoment_0"
F_HU_MOMENT_1 = "HuMoment_1"
F_HU_MOMENT_2 = "HuMoment_2"
F_HU_MOMENT_3 = "HuMoment_3"
F_HU_MOMENT_4 = "HuMoment_4"
F_HU_MOMENT_5 = "HuMoment_5"
F_HU_MOMENT_6 = "HuMoment_6"
F_INERTIA_TENSOR_0_0 = "InertiaTensor_0_0"
F_INERTIA_TENSOR_0_1 = "InertiaTensor_0_1"
F_INERTIA_TENSOR_1_0 = "InertiaTensor_1_0"
F_INERTIA_TENSOR_1_1 = "InertiaTensor_1_1"
F_INERTIA_TENSOR_EIGENVALUES_0 = "InertiaTensorEigenvalues_0"
F_INERTIA_TENSOR_EIGENVALUES_1 = "InertiaTensorEigenvalues_1"
F_NORMALIZED_MOMENT_0_0 = "NormalizedMoment_0_0"
F_NORMALIZED_MOMENT_0_1 = "NormalizedMoment_0_1"
F_NORMALIZED_MOMENT_0_2 = "NormalizedMoment_0_2"
F_NORMALIZED_MOMENT_0_3 = "NormalizedMoment_0_3"
F_NORMALIZED_MOMENT_1_0 = "NormalizedMoment_1_0"
F_NORMALIZED_MOMENT_1_1 = "NormalizedMoment_1_1"
F_NORMALIZED_MOMENT_1_2 = "NormalizedMoment_1_2"
F_NORMALIZED_MOMENT_1_3 = "NormalizedMoment_1_3"
F_NORMALIZED_MOMENT_2_0 = "NormalizedMoment_2_0"
F_NORMALIZED_MOMENT_2_1 = "NormalizedMoment_2_1"
F_NORMALIZED_MOMENT_2_2 = "NormalizedMoment_2_2"
F_NORMALIZED_MOMENT_2_3 = "NormalizedMoment_2_3"
F_NORMALIZED_MOMENT_3_0 = "NormalizedMoment_3_0"
F_NORMALIZED_MOMENT_3_1 = "NormalizedMoment_3_1"
F_NORMALIZED_MOMENT_3_2 = "NormalizedMoment_3_2"
F_NORMALIZED_MOMENT_3_3 = "NormalizedMoment_3_3"
F_SPATIAL_MOMENT_0_0 = "SpatialMoment_0_0"
F_SPATIAL_MOMENT_0_1 = "SpatialMoment_0_1"
F_SPATIAL_MOMENT_0_2 = "SpatialMoment_0_2"
F_SPATIAL_MOMENT_0_3 = "SpatialMoment_0_3"
F_SPATIAL_MOMENT_1_0 = "SpatialMoment_1_0"
F_SPATIAL_MOMENT_1_1 = "SpatialMoment_1_1"
F_SPATIAL_MOMENT_1_2 = "SpatialMoment_1_2"
F_SPATIAL_MOMENT_1_3 = "SpatialMoment_1_3"
F_SPATIAL_MOMENT_2_0 = "SpatialMoment_2_0"
F_SPATIAL_MOMENT_2_1 = "SpatialMoment_2_1"
F_SPATIAL_MOMENT_2_2 = "SpatialMoment_2_2"
F_SPATIAL_MOMENT_2_3 = "SpatialMoment_2_3"

"""The non-Zernike features"""
F_STD_2D = [
    F_AREA,
    F_PERIMETER,
    F_MAXIMUM_RADIUS,
    F_MEAN_RADIUS,
    F_MEDIAN_RADIUS,
    F_MIN_FERET_DIAMETER,
    F_MAX_FERET_DIAMETER,
    F_ORIENTATION,
    F_ECCENTRICITY,
    F_FORM_FACTOR,
    F_SOLIDITY,
    F_CONVEX_AREA,
    F_COMPACTNESS,
    F_BBOX_AREA,
    F_CELL_LEN,
    F_CELL_WIDTH,
]
F_STD_3D = [
    F_VOLUME,
    F_SURFACE_AREA,
    F_CENTER_Z,
    F_BBOX_VOLUME,
    F_MIN_Z,
    F_MAX_Z,
]
F_ADV_2D = [
    F_SPATIAL_MOMENT_0_0,
    F_SPATIAL_MOMENT_0_1,
    F_SPATIAL_MOMENT_0_2,
    F_SPATIAL_MOMENT_0_3,
    F_SPATIAL_MOMENT_1_0,
    F_SPATIAL_MOMENT_1_1,
    F_SPATIAL_MOMENT_1_2,
    F_SPATIAL_MOMENT_1_3,
    F_SPATIAL_MOMENT_2_0,
    F_SPATIAL_MOMENT_2_1,
    F_SPATIAL_MOMENT_2_2,
    F_SPATIAL_MOMENT_2_3,
    F_CENTRAL_MOMENT_0_0,
    F_CENTRAL_MOMENT_0_1,
    F_CENTRAL_MOMENT_0_2,
    F_CENTRAL_MOMENT_0_3,
    F_CENTRAL_MOMENT_1_0,
    F_CENTRAL_MOMENT_1_1,
    F_CENTRAL_MOMENT_1_2,
    F_CENTRAL_MOMENT_1_3,
    F_CENTRAL_MOMENT_2_0,
    F_CENTRAL_MOMENT_2_1,
    F_CENTRAL_MOMENT_2_2,
    F_CENTRAL_MOMENT_2_3,
    F_NORMALIZED_MOMENT_0_0,
    F_NORMALIZED_MOMENT_0_1,
    F_NORMALIZED_MOMENT_0_2,
    F_NORMALIZED_MOMENT_0_3,
    F_NORMALIZED_MOMENT_1_0,
    F_NORMALIZED_MOMENT_1_1,
    F_NORMALIZED_MOMENT_1_2,
    F_NORMALIZED_MOMENT_1_3,
    F_NORMALIZED_MOMENT_2_0,
    F_NORMALIZED_MOMENT_2_1,
    F_NORMALIZED_MOMENT_2_2,
    F_NORMALIZED_MOMENT_2_3,
    F_NORMALIZED_MOMENT_3_0,
    F_NORMALIZED_MOMENT_3_1,
    F_NORMALIZED_MOMENT_3_2,
    F_NORMALIZED_MOMENT_3_3,
    F_HU_MOMENT_0,
    F_HU_MOMENT_1,
    F_HU_MOMENT_2,
    F_HU_MOMENT_3,
    F_HU_MOMENT_4,
    F_HU_MOMENT_5,
    F_HU_MOMENT_6,
    F_INERTIA_TENSOR_0_0,
    F_INERTIA_TENSOR_0_1,
    F_INERTIA_TENSOR_1_0,
    F_INERTIA_TENSOR_1_1,
    F_INERTIA_TENSOR_EIGENVALUES_0,
    F_INERTIA_TENSOR_EIGENVALUES_1,
]
F_ADV_3D = [F_SOLIDITY]
F_STANDARD = [
    F_EXTENT,
    F_EULER_NUMBER,
    F_EQUIVALENT_DIAMETER,
    F_MAJOR_AXIS_LENGTH,
    F_MINOR_AXIS_LENGTH,
    F_CENTER_X,
    F_CENTER_Y,
    F_MIN_X,
    F_MIN_Y,
    F_MAX_X,
    F_MAX_Y,
]


class MeasureCelllength(Module):
    module_name = "MeasureCelllength"
    variable_revision_number = 3
    category = "Measurement"

    def create_settings(self):
        """Create the settings for the module at startup and set the module name

        The module allows for an unlimited number of measured objects, each
        of which has an entry in self.object_groups.
        """
        self.objects_list = LabelListSubscriber(
            "Select object sets to measure",
            [],
            doc="""Select the object sets whose size and shape you want to measure.""",
        )
        self.spacer = Divider(line=True)

        self.calculate_advanced = Binary(
            text="Calculate the advanced features?",
            value=False,
            doc="""\
Select *{YES}* to calculate additional statistics for object moments
and intertia tensors in **2D mode**. These features should not require much additional time
to calculate, but do add many additional columns to the resulting output 
files.

In **3D mode** this setting enables the Solidity measurement, which can be time-consuming
to calculate.""".format(
                **{"YES": "Yes"}
            ),
        )

        self.calculate_zernikes = Binary(
            text="Calculate the Zernike features?",
            value=True,
            doc="""\
Select *{YES}* to calculate the Zernike shape features. Because the
first 10 Zernike polynomials (from order 0 to order 9) are calculated,
this operation can be time consuming if the image contains a lot of
objects. Select *{NO}* if you are measuring 3D objects with this
module.""".format(
                **{"YES": "Yes", "NO": "No"}
            ),
        )

    def settings(self):
        """The settings as they appear in the save file"""
        result = [self.objects_list, self.calculate_zernikes, self.calculate_advanced]
        return result

    def visible_settings(self):
        """The settings as they appear in the module viewer"""
        result = [
            self.objects_list,
            self.spacer,
            self.calculate_zernikes,
            self.calculate_advanced,
        ]
        return result

    def validate_module(self, pipeline):
        """Make sure chosen objects are selected only once"""
        objects = set()
        if len(self.objects_list.value) == 0:
            raise ValidationError("No object sets selected", self.objects_list)

        for object_name in self.objects_list.value:
            if object_name in objects:
                raise ValidationError(
                    "%s has already been selected" % object_name, object_name
                )
            objects.add(object_name)

    def get_categories(self, pipeline, object_name):
        """Get the categories of measurements supplied for the given object name

        pipeline - pipeline being run
        object_name - name of labels in question (or 'Images')
        returns a list of category names
        """
        for object_set in self.objects_list.value:
            if object_set == object_name:
                return [AREA_SHAPE]
        else:
            return []

    def get_zernike_numbers(self):
        """The Zernike numbers measured by this module"""
        if self.calculate_zernikes.value:
            return centrosome.zernike.get_zernike_indexes(ZERNIKE_N + 1)
        else:
            return []

    def get_zernike_name(self, zernike_index):
        """Return the name of a Zernike feature, given a (N,M) 2-tuple

        zernike_index - a 2 element sequence organized as N,M
        """
        return "Zernike_%d_%d" % (zernike_index[0], zernike_index[1])

    def get_feature_names(self, pipeline):
        """Return the names of the features measured"""
        feature_names = list(F_STANDARD)

        if pipeline.volumetric():
            feature_names += list(F_STD_3D)
            if self.calculate_advanced.value:
                feature_names += list(F_ADV_3D)
        else:
            feature_names += list(F_STD_2D)
            if self.calculate_zernikes.value:
                feature_names += [
                    self.get_zernike_name(index) for index in self.get_zernike_numbers()
                ]
            if self.calculate_advanced.value:
                feature_names += list(F_ADV_2D)

        return feature_names

    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces

        object_name - return measurements made on this object
                      (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        if category == AREA_SHAPE and self.get_categories(pipeline, object_name):
            return self.get_feature_names(pipeline)
        return []

    def run(self, workspace):
        """Run, computing the area measurements for the objects"""

        if self.show_window:
            workspace.display_data.col_labels = (
                "Object",
                "Feature",
                "Mean",
                "Median",
                "STD",
            )

            workspace.display_data.statistics = []
        for object_name in self.objects_list.value:
            self.run_on_objects(object_name, workspace)

    def run_on_objects(self, object_name, workspace):
        """Determine desired measurements and pass in object arrays for analysis"""
        objects = workspace.get_objects(object_name)

        # Don't analyze if there are no objects at all.

        if len(objects.indices) == 0:
            # No objects to process
            self.measurements_without_objects(workspace, object_name)
            return

        # Determine which properties we're measuring.
        if len(objects.shape) == 2:
            desired_properties = [
                "label",
                "image",
                "area",
                "perimeter",
                "bbox",
                "bbox_area",
                "major_axis_length",
                "minor_axis_length",
                "orientation",
                "centroid",
                "equivalent_diameter",
                "extent",
                "eccentricity",
                "convex_area",
                "solidity",
                "euler_number",

            ]
            if self.calculate_advanced.value:
                desired_properties += [
                    "inertia_tensor",
                    "inertia_tensor_eigvals",
                    "moments",
                    "moments_central",
                    "moments_hu",
                    "moments_normalized",
                ]
        else:
            desired_properties = [
                "label",
                "image",
                "area",
                "centroid",
                "bbox",
                "bbox_area",
                "major_axis_length",
                "minor_axis_length",
                "extent",
                "equivalent_diameter",
                "euler_number",
            ]
            if self.calculate_advanced.value:
                desired_properties += [
                    "solidity",
                ]

        # Check for overlapping object sets
        if not objects.overlapping():
            features_to_record = self.analyze_objects(objects, desired_properties)
        else:
            # Objects are overlapping, process as single arrays
            coords_array = objects.ijv
            features_to_record = {}
            for label in objects.indices:
                omap = numpy.zeros(objects.shape)
                ocoords = coords_array[coords_array[:, 2] == label, 0:2]
                numpy.put(omap, numpy.ravel_multi_index(ocoords.T, omap.shape), 1)
                tempobject = Objects()
                tempobject.segmented = omap
                buffer = self.analyze_objects(tempobject, desired_properties)
                for f, m in buffer.items():
                    if f in features_to_record:
                        features_to_record[f] = numpy.concatenate(
                            (features_to_record[f], m)
                        )
                    else:
                        features_to_record[f] = m
        for f, m in features_to_record.items():
            self.record_measurement(workspace, object_name, f, m)

    def analyze_objects(self, objects, desired_properties):
        """Computing the measurements for a single map of objects"""
        labels = objects.segmented
        nobjects = len(objects.indices)
        if len(objects.shape) == 2:
            props = skimage.measure.regionprops_table(
                labels, properties=desired_properties
            )

            formfactor = 4.0 * numpy.pi * props["area"] / props["perimeter"] ** 2
            denom = [max(x, 1) for x in 4.0 * numpy.pi * props["area"]]
            compactness = props["perimeter"] ** 2 / denom
            Cell_length = props["bbox-3"] - props["bbox-1"]
            Cell_width = props["bbox-2"] - props["bbox-0"]
            formfactor = props["perimeter"] ** 2 / (4 * numpy.pi * props["area"])
            compactness = props["perimeter"] ** 2 / (4 * numpy.pi * props["area"])
            max_radius = numpy.zeros(nobjects)
            median_radius = numpy.zeros(nobjects)
            mean_radius = numpy.zeros(nobjects)
            min_feret_diameter = numpy.zeros(nobjects)
            max_feret_diameter = numpy.zeros(nobjects)
            zernike_numbers = self.get_zernike_numbers()

            zf = {}
            for n, m in zernike_numbers:
                zf[(n, m)] = numpy.zeros(nobjects)

            for index, mini_image in enumerate(props["image"]):
                # Pad image to assist distance tranform
                mini_image = numpy.pad(mini_image, 1)
                distances = scipy.ndimage.distance_transform_edt(mini_image)
                max_radius[index] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                    scipy.ndimage.maximum(distances, mini_image)
                )
                mean_radius[index] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                    scipy.ndimage.mean(distances, mini_image)
                )
                median_radius[index] = centrosome.cpmorphology.median_of_labels(
                    distances, mini_image.astype("int"), [1]
                )
            #
            # Zernike features
            #
            if self.calculate_zernikes.value:
                zf_l = centrosome.zernike.zernike(
                    zernike_numbers, labels, objects.indices
                )
                for (n, m), z in zip(zernike_numbers, zf_l.transpose()):
                    zf[(n, m)] = z

            if nobjects > 0:
                chulls, chull_counts = centrosome.cpmorphology.convex_hull_ijv(
                    objects.ijv, objects.indices
                )
                #
                # Feret diameter
                #
                (
                    min_feret_diameter,
                    max_feret_diameter,
                ) = centrosome.cpmorphology.feret_diameter(
                    chulls, chull_counts, objects.indices
                )

            features_to_record = {
                F_AREA: props["area"],
                F_PERIMETER: props["perimeter"],
                F_MAJOR_AXIS_LENGTH: props["major_axis_length"],
                F_MINOR_AXIS_LENGTH: props["minor_axis_length"],
                F_ECCENTRICITY: props["eccentricity"],
                F_ORIENTATION: props["orientation"] * (180 / numpy.pi),
                F_CENTER_X: props["centroid-1"],
                F_CENTER_Y: props["centroid-0"],
                F_BBOX_AREA: props["bbox_area"],
                F_CELL_WIDTH: Cell_width,
                F_CELL_LEN: Cell_length,
                F_MIN_X: props["bbox-1"],
                F_MAX_X: props["bbox-3"],
                F_MIN_Y: props["bbox-0"],
                F_MAX_Y: props["bbox-2"],
                F_FORM_FACTOR: formfactor,
                F_EXTENT: props["extent"],
                F_SOLIDITY: props["solidity"],
                F_COMPACTNESS: compactness,
                F_EULER_NUMBER: props["euler_number"],
                F_MAXIMUM_RADIUS: max_radius,
                F_MEAN_RADIUS: mean_radius,
                F_MEDIAN_RADIUS: median_radius,
                F_CONVEX_AREA: props["convex_area"],
                F_MIN_FERET_DIAMETER: min_feret_diameter,
                F_MAX_FERET_DIAMETER: max_feret_diameter,
                F_EQUIVALENT_DIAMETER: props["equivalent_diameter"],
            }
            if self.calculate_advanced.value:
                features_to_record.update(
                    {
                        F_SPATIAL_MOMENT_0_0: props["moments-0-0"],
                        F_SPATIAL_MOMENT_0_1: props["moments-0-1"],
                        F_SPATIAL_MOMENT_0_2: props["moments-0-2"],
                        F_SPATIAL_MOMENT_0_3: props["moments-0-3"],
                        F_SPATIAL_MOMENT_1_0: props["moments-1-0"],
                        F_SPATIAL_MOMENT_1_1: props["moments-1-1"],
                        F_SPATIAL_MOMENT_1_2: props["moments-1-2"],
                        F_SPATIAL_MOMENT_1_3: props["moments-1-3"],
                        F_SPATIAL_MOMENT_2_0: props["moments-2-0"],
                        F_SPATIAL_MOMENT_2_1: props["moments-2-1"],
                        F_SPATIAL_MOMENT_2_2: props["moments-2-2"],
                        F_SPATIAL_MOMENT_2_3: props["moments-2-3"],
                        F_CENTRAL_MOMENT_0_0: props["moments_central-0-0"],
                        F_CENTRAL_MOMENT_0_1: props["moments_central-0-1"],
                        F_CENTRAL_MOMENT_0_2: props["moments_central-0-2"],
                        F_CENTRAL_MOMENT_0_3: props["moments_central-0-3"],
                        F_CENTRAL_MOMENT_1_0: props["moments_central-1-0"],
                        F_CENTRAL_MOMENT_1_1: props["moments_central-1-1"],
                        F_CENTRAL_MOMENT_1_2: props["moments_central-1-2"],
                        F_CENTRAL_MOMENT_1_3: props["moments_central-1-3"],
                        F_CENTRAL_MOMENT_2_0: props["moments_central-2-0"],
                        F_CENTRAL_MOMENT_2_1: props["moments_central-2-1"],
                        F_CENTRAL_MOMENT_2_2: props["moments_central-2-2"],
                        F_CENTRAL_MOMENT_2_3: props["moments_central-2-3"],
                        F_NORMALIZED_MOMENT_0_0: props["moments_normalized-0-0"],
                        F_NORMALIZED_MOMENT_0_1: props["moments_normalized-0-1"],
                        F_NORMALIZED_MOMENT_0_2: props["moments_normalized-0-2"],
                        F_NORMALIZED_MOMENT_0_3: props["moments_normalized-0-3"],
                        F_NORMALIZED_MOMENT_1_0: props["moments_normalized-1-0"],
                        F_NORMALIZED_MOMENT_1_1: props["moments_normalized-1-1"],
                        F_NORMALIZED_MOMENT_1_2: props["moments_normalized-1-2"],
                        F_NORMALIZED_MOMENT_1_3: props["moments_normalized-1-3"],
                        F_NORMALIZED_MOMENT_2_0: props["moments_normalized-2-0"],
                        F_NORMALIZED_MOMENT_2_1: props["moments_normalized-2-1"],
                        F_NORMALIZED_MOMENT_2_2: props["moments_normalized-2-2"],
                        F_NORMALIZED_MOMENT_2_3: props["moments_normalized-2-3"],
                        F_NORMALIZED_MOMENT_3_0: props["moments_normalized-3-0"],
                        F_NORMALIZED_MOMENT_3_1: props["moments_normalized-3-1"],
                        F_NORMALIZED_MOMENT_3_2: props["moments_normalized-3-2"],
                        F_NORMALIZED_MOMENT_3_3: props["moments_normalized-3-3"],
                        F_HU_MOMENT_0: props["moments_hu-0"],
                        F_HU_MOMENT_1: props["moments_hu-1"],
                        F_HU_MOMENT_2: props["moments_hu-2"],
                        F_HU_MOMENT_3: props["moments_hu-3"],
                        F_HU_MOMENT_4: props["moments_hu-4"],
                        F_HU_MOMENT_5: props["moments_hu-5"],
                        F_HU_MOMENT_6: props["moments_hu-6"],
                        F_INERTIA_TENSOR_0_0: props["inertia_tensor-0-0"],
                        F_INERTIA_TENSOR_0_1: props["inertia_tensor-0-1"],
                        F_INERTIA_TENSOR_1_0: props["inertia_tensor-1-0"],
                        F_INERTIA_TENSOR_1_1: props["inertia_tensor-1-1"],
                        F_INERTIA_TENSOR_EIGENVALUES_0: props[
                            "inertia_tensor_eigvals-0"
                        ],
                        F_INERTIA_TENSOR_EIGENVALUES_1: props[
                            "inertia_tensor_eigvals-1"
                        ],
                    }
                )

            if self.calculate_zernikes.value:
                features_to_record.update(
                    {
                        self.get_zernike_name((n, m)): zf[(n, m)]
                        for n, m in zernike_numbers
                    }
                )

        else:

            props = skimage.measure.regionprops_table(
                labels, properties=desired_properties
            )

            # SurfaceArea
            surface_areas = numpy.zeros(len(props["label"]))
            for index, label in enumerate(props["label"]):
                # this seems less elegant than you might wish, given that regionprops returns a slice,
                # but we need to expand the slice out by one voxel in each direction, or surface area freaks out
                volume= labels[max(props['bbox-0'][index]-1,0):min(props['bbox-3'][index]+1,labels.shape[0]),
                          max(props['bbox-1'][index]-1,0):min(props['bbox-4'][index]+1,labels.shape[1]),
                          max(props['bbox-2'][index]-1,0):min(props['bbox-5'][index]+1,labels.shape[2])] 
                volume = volume == label
                verts, faces, _normals, _values = skimage.measure.marching_cubes(
                    volume,
                    method="lewiner",
                    spacing=objects.parent_image.spacing
                    if objects.has_parent_image
                    else (1.0,) * labels.ndim,
                    level=0,
                )
                surface_areas[index] = skimage.measure.mesh_surface_area(verts, faces)

            features_to_record = {
                F_VOLUME: props["area"],
                F_SURFACE_AREA: surface_areas,
                F_MAJOR_AXIS_LENGTH: props["major_axis_length"],
                F_MINOR_AXIS_LENGTH: props["minor_axis_length"],
                F_CENTER_X: props["centroid-2"],
                F_CENTER_Y: props["centroid-1"],
                F_CENTER_Z: props["centroid-0"],
                F_BBOX_VOLUME: props["bbox_area"],
                F_MIN_X: props["bbox-2"],
                F_MAX_X: props["bbox-5"],
                F_MIN_Y: props["bbox-1"],
                F_MAX_Y: props["bbox-4"],
                F_MIN_Z: props["bbox-0"],
                F_MAX_Z: props["bbox-3"],
                F_EXTENT: props["extent"],
                F_EULER_NUMBER: props["euler_number"],
                F_EQUIVALENT_DIAMETER: props["equivalent_diameter"],
            }
            if self.calculate_advanced.value:
                features_to_record[F_SOLIDITY] = props["solidity"]
        return features_to_record

    def display(self, workspace, figure):
        figure.set_subplots((1, 1))
        figure.subplot_table(
            0,
            0,
            workspace.display_data.statistics,
            col_labels=workspace.display_data.col_labels,
            title="default",
        )

    def perform_measurement(self, workspace, function, object_name, feature_name):
        """Perform a measurement on a label matrix

        workspace   - the workspace for the run
        function    - a function with the following sort of signature:
                      image - an image to be fed into the function which for
                              our case is all ones
                      labels - the label matrix from the objects
                      index  - a sequence of label indexes to pay attention to
        object_name - name of object to retrieve from workspace and deposit
                      in measurements
        feature_name- name of feature to deposit in measurements
        """
        objects = workspace.get_objects(object_name)
        if len(objects.indices) > 0:
            data = objects.fn_of_label_and_index(function)
        else:
            data = numpy.zeros((0,))
        self.record_measurement(workspace, object_name, feature_name, data)

    def perform_ndmeasurement(self, workspace, function, object_name, feature_name):
        """Perform a scipy.ndimage-style measurement on a label matrix

        workspace   - the workspace for the run
        function    - a function with the following sort of signature:
                      image - an image to be fed into the function which for
                              our case is all ones
                      labels - the label matrix from the objects
                      index  - a sequence of label indexes to pay attention to
        object_name - name of object to retrieve from workspace and deposit
                      in measurements
        feature_name- name of feature to deposit in measurements
        """
        objects = workspace.get_objects(object_name)
        if len(objects.indices) > 0:
            data = objects.fn_of_ones_label_and_index(function)
        else:
            data = numpy.zeros((0,))
        self.record_measurement(workspace, object_name, feature_name, data)

    def record_measurement(self, workspace, object_name, feature_name, result):
        """Record the result of a measurement in the workspace's measurements"""
        data = centrosome.cpmorphology.fixup_scipy_ndimage_result(result)
        workspace.add_measurement(
            object_name, "%s_%s" % (AREA_SHAPE, feature_name), data
        )
        if self.show_window and numpy.any(numpy.isfinite(data)) > 0:
            data = data[numpy.isfinite(data)]
            workspace.display_data.statistics.append(
                (
                    object_name,
                    feature_name,
                    "%.2f" % numpy.mean(data),
                    "%.2f" % numpy.median(data),
                    "%.2f" % numpy.std(data),
                )
            )

    def get_measurement_columns(self, pipeline):
        """Return measurement column definitions.
        All cols returned as float even though "Area" will only ever be int"""
        measurement_names = self.get_feature_names(pipeline)
        cols = []
        for oname in self.objects_list.value:
            for mname in measurement_names:
                cols += [(oname, AREA_SHAPE + "_" + mname, COLTYPE_FLOAT,)]
        return cols

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        """Adjust the setting_values for older save file versions"""
        if variable_revision_number == 1:
            objects_list = setting_values[:-1]
            setting_values = [", ".join(map(str, objects_list)), setting_values[-1]]
            variable_revision_number = 2
        if variable_revision_number == 2:
            # Add advanced features toggle
            setting_values.append("No")
            variable_revision_number = 3
        return setting_values, variable_revision_number

    def volumetric(self):
        return True

    def measurements_without_objects(self, workspace, object_name):
        # Create column headers even if there were no objects in a set.
        features_to_record = self.get_feature_names(workspace.pipeline)
        empty_measure = numpy.zeros((0,))
        for feature_name in features_to_record:
            self.record_measurement(workspace, object_name, feature_name, empty_measure)
