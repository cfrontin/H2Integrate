from attrs import field, define

import numpy as np

from h2integrate.core.utilities import merge_shared_inputs
from h2integrate.core.validators import gt_zero, gte_zero, range_val
from h2integrate.core.model_baseclasses import CostModelBaseClass, CostModelBaseConfig


@define(kw_only=True)
class EnhancedATBWindPlantCostModelConfig(CostModelBaseConfig):
    """Configuration class for the ATBWindCostModel.
    Recommended to use with wind models (Land-Based, Offshore and Distributed
    More information on ATB methodology and representative wind technologies can
    be found `here <https://atb.nrel.gov/electricity/2024/technologies>`_
    Reference cost values can be found on the `Land-Based Wind`,
    `Fixed-Bottom Offshore Wind`, `Floating Offshore Wind` or `Distributed Wind`
    sheet of the `NREL ATB workbook <https://atb.nrel.gov/electricity/2024/data>`_.

    Attributes:
        capex_per_kW (float|int): capital cost of wind system in $/kW
        opex_per_kW_per_year (float|int): annual operating cost of wind
            system in $/kW/year
    """

    capex_per_kW: float | int = field(validator=gte_zero)
    opex_per_kW_per_year: float | int = field(validator=gte_zero)
    BOS_fraction: float = field(validator=range_val(0.0, 1.0))
    hub_height: float = field(validator=gt_zero)
    rotor_diameter: float = field(validator=gt_zero)
    turbine_rating_kw: float = field(validator=gt_zero)


class EnhancedATBWindPlantCostModel(CostModelBaseClass):
    """
    OpenMDAO component for calculating wind plant capital and operating expenditures.

    This component calculates the capital expenditure (CapEx) and annual operating
    expenditure (OpEx) of a wind plant based on its rated capacity and cost model
    parameters defined in an `EnhancedATBWindPlantCostModelConfig`.

    Attributes:
        config (EnhancedATBWindPlantCostModelConfig):
            Configuration object containing per-kW cost parameters for CapEx and OpEx.

    Inputs:
        rated_electricity_production (float):
            Rated capacity of the wind farm [kW].

    Outputs:
        CapEx (float):
            Total capital expenditure of the wind plant.
        OpEx (float):
            Annual operating expenditure of the wind plant.
    """

    _time_step_bounds = (
        3600,
        3600,
    )  # (min, max) time step lengths (in seconds) compatible with this model

    def setup(self):
        self.config = EnhancedATBWindPlantCostModelConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "cost"),
            additional_cls_name=self.__class__.__name__,
        )
        super().setup()

        self.add_input(
            "rated_electricity_production",
            val=0.0,
            units="kW",
            desc="Wind farm rated capacity in kW",
        )

        self.add_input(
            "wind_turbine_rating",
            val=self.config.turbine_rating_kw,
            units="kW",
            desc="rating of an individual turbine in kW",
        )

        self.add_input(
            "hub_height",
            val=self.config.hub_height,
            units="m",
            desc="turbine hub height",
        )

        self.add_input(
            "rotor_diameter",
            val=self.config.rotor_diameter,
            units="m",
            desc="turbine rotor diameter",
        )

        self.add_output(
            "specific_power",
            val=0.0,
            units="W/m**2",
            desc="specific power of the rotor",
        )

        self.add_output(
            "tip_clearance",
            val=0.0,
            units="m",
            desc="rotor tip path ground clearance",
        )


    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        capex_orig = self.config.capex_per_kW * inputs["rated_electricity_production"]
        # print(f"capex_orig: {capex_orig}")  # DEBUG!!!!!
        capex_turbine_base = 6.164  # MUSD,
        SP_base = 232.5  # -, from: https://atb.nrel.gov/electricity/2024b/land-based_wind
        P_base = 5900.0  # kW, from: https://atb.nrel.gov/electricity/2024b/land-based_wind
        D_base = np.sqrt(4 / np.pi * (inputs["wind_turbine_rating"] * 1000.0) / SP_base)

        # this is based on a silly analysis in @cfrontin's h2i_sandbox/csm_sandbox.ipynb
        p_slopes = [0.02209403, 0.1357578, 0.27937119]
        p_intercepts = [ 0.69012541, 1.00728755, -0.19892396]
        # those are based on
        # print(f"P_base: {P_base}; inputs['turbine_rating_kw']: {inputs['turbine_rating_kw']}")  # DEBUG!!!!!
        # print(f"D_base: {D_base}; inputs['rotor_diameter']: {inputs['rotor_diameter']}")  # DEBUG!!!!!
        slope_rotor_diameter = np.polyval(p_slopes, (inputs["wind_turbine_rating"] - P_base)/P_base)
        intercept_rotor_diameter = np.polyval(p_intercepts, (inputs["wind_turbine_rating"] - P_base)/P_base)
        rotor_diameter_capex_adjustment = slope_rotor_diameter*(inputs["rotor_diameter"] - D_base)/D_base + intercept_rotor_diameter
        # print("rotor_diameter_capex_adjustment:", rotor_diameter_capex_adjustment)  # DEBUG!!!!!

        capex = (
            self.config.BOS_fraction * capex_orig
            + (1 + rotor_diameter_capex_adjustment) * (1.0 - self.config.BOS_fraction) * capex_orig
        )
        opex = self.config.opex_per_kW_per_year * inputs["rated_electricity_production"]

        outputs["tip_clearance"] = inputs["hub_height"] - 0.5*inputs["rotor_diameter"]
        outputs["specific_power"] = inputs["wind_turbine_rating"]*1000.0/(0.25*np.pi*inputs["rotor_diameter"]**2)

        outputs["CapEx"] = capex
        outputs["OpEx"] = opex