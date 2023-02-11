import numpy as np

from ortools.sat.python import cp_model
from typing import List, Dict


FLOAT_APPROX_PRECISION = 100


def create_boolean_is_positive(
    model: cp_model.CpModel, var: cp_model.IntVar, big_m=1000
):
    """Create a bool variable such that
    If var >= 0 then bool = 1
    If var  < 0 then bool = 0

    Big M should be larger than x largest absolute value
    """

    boolean_var = model.NewBoolVar(name=var.Name() + "_is_positive")

    # Bool are casted to 0 if False and 1 if True, so you can do some operations with them

    # If var > 0, then this is true only if bool = 1
    # If var <= 0, then this is always true
    model.Add(big_m * boolean_var >= var)
    # If var > 0, then this is always true
    # If var <= 0, then this is true only if bool = 0
    model.Add(big_m * (boolean_var - 1) <= var)

    # To handle the case var == 0, we specifiy that we want boolean_var to be = 1 this way
    model.Add(boolean_var > var).OnlyEnforceIf(boolean_var.Not())

    return boolean_var


def create_boolean_is_equal_to(
    model: cp_model.CpModel, var: cp_model.IntVar, value: int
):
    """Create a bool variable such that
    If var == value then bool = 1
    Else then bool = 0
    """
    boolean_var = model.NewBoolVar(name=f"{var.Name()}_is_equal_to_{value}")
    model.Add(value * boolean_var == var).OnlyEnforceIf(boolean_var)
    model.Add(value != var).OnlyEnforceIf(boolean_var.Not())

    return boolean_var


def add_multiplication_constraint(
    model: cp_model.CpModel,
    target: cp_model.IntVar,
    variables: List[cp_model.IntVar],
):
    if len(variables) <= 2:
        # If less than 2 variables, we can add a normal inequality constraint
        model.AddMultiplicationEquality(target, variables)
        return
    else:
        last_variable = variables.pop()
        before_last_variable = variables.pop()
        # Use their bounds to define domain of the intermediate variable
        # You may need additional logic here to account for variable domain bounds
        ub = max(
            last_variable.Proto().domain[1] * before_last_variable.Proto().domain[1],
            last_variable.Proto().domain[0] * before_last_variable.Proto().domain[1],
            last_variable.Proto().domain[0] * before_last_variable.Proto().domain[0],
            last_variable.Proto().domain[1] * before_last_variable.Proto().domain[0],
        )
        lb = min(
            last_variable.Proto().domain[1] * before_last_variable.Proto().domain[1],
            last_variable.Proto().domain[0] * before_last_variable.Proto().domain[1],
            last_variable.Proto().domain[0] * before_last_variable.Proto().domain[0],
            last_variable.Proto().domain[1] * before_last_variable.Proto().domain[0],
        )
        # Create an intermediate variable
        intermediate = model.NewIntVar(
            lb=lb, ub=ub, name=f"{before_last_variable.Name()}*{last_variable.Name()}"
        )
        model.AddMultiplicationEquality(
            intermediate, [before_last_variable, last_variable]
        )
        # Recursion
        add_multiplication_constraint(model, target, variables + [intermediate])


def create_polynom(
    model: cp_model.CpModel,
    var: cp_model.IntVar,
    coefs: List[float],
    float_precision=FLOAT_APPROX_PRECISION,
    verbose=True,
):
    """This polynom takes as an input var a usual integer"""
    degree = len(coefs)

    # Approximate all coef values by mutliplying them by a big number and rounding them
    coefs = [round(float_precision * coef) for coef in coefs]

    polynom_value = 0
    polynom_value_ub = 0
    polynom_value_lb = 0
    for deg in range(degree):
        # Create the coefficient value
        if deg == 0:
            polynom_value += coefs[deg]
            polynom_value_lb += coefs[deg]
            polynom_value_ub += coefs[deg]
        elif deg == 1:
            polynom_value_lb += var.Proto().domain[0] * coefs[deg]
            polynom_value_ub += var.Proto().domain[1] * coefs[deg]

            polynom_value += coefs[deg] * var
        else:
            lb = var.Proto().domain[0] ** deg
            ub = var.Proto().domain[1] ** deg

            if (deg % 2) == 0:
                if var.Proto().domain[0] < 0:
                    lb = -lb
                if var.Proto().domain[1] < 0:
                    ub = -ub

            lb = coefs[deg] * lb
            ub = coefs[deg] * ub

            polynom_value_lb += lb
            polynom_value_ub += ub

            target = model.NewIntVar(lb=lb, ub=ub, name=f"{var.Name()}**{deg}")
            add_multiplication_constraint(model, target, [var] * deg)
            polynom_value += coefs[deg] * target

    if verbose:
        print("Polynom", polynom_value)

    polynom_var = model.NewIntVar(
        polynom_value_lb, polynom_value_ub, name=f"{var.Name()}_polynom"
    )
    model.Add(polynom_var == polynom_value)
    return polynom_var


def create_polynom_decimal(
    model: cp_model.CpModel,
    var: cp_model.IntVar,
    coefs: List[float],
    float_precision_var=FLOAT_APPROX_PRECISION,
    float_precision_coef=FLOAT_APPROX_PRECISION,
    verbose=True,
):
    """This polynom accepts as an input a decimal var upscaled by float_precision_var."""
    degree = len(coefs)

    # Approximate all coef values by mutliplying them by a big number and rounding them
    coefs = [round(float_precision_coef * coef) for coef in coefs]

    polynom_value = 0
    polynom_value_ub = 0
    polynom_value_lb = 0
    for deg in range(degree):
        # Create the coefficient value
        if deg == 0:
            polynom_value += coefs[deg] * float_precision_var
            polynom_value_lb += coefs[deg] * float_precision_var
            polynom_value_ub += coefs[deg] * float_precision_var
        elif deg == 1:
            polynom_value_lb += var.Proto().domain[0] * coefs[deg]
            polynom_value_ub += var.Proto().domain[1] * coefs[deg]

            polynom_value += coefs[deg] * var
        else:
            # Bounds logic
            lb_no_coef = var.Proto().domain[0] ** deg
            ub_no_coef = var.Proto().domain[1] ** deg

            if (deg % 2) == 0:
                if var.Proto().domain[0] < 0:
                    lb_no_coef = -lb_no_coef
                if var.Proto().domain[1] < 0:
                    ub_no_coef = -ub_no_coef

            lb = coefs[deg] * lb_no_coef
            ub = coefs[deg] * ub_no_coef

            polynom_value_lb += lb
            polynom_value_ub += ub

            # Compute (x**n)
            target = model.NewIntVar(
                lb=lb_no_coef, ub=ub_no_coef, name=f"{var.Name()}**{deg}"
            )
            add_multiplication_constraint(model, target, [var] * deg)
            # Then compute (a * x**n)
            target_times_coef = model.NewIntVar(
                lb=lb, ub=ub, name=f"{coefs[deg]}*{var.Name()}**{deg}"
            )
            model.Add(target_times_coef == target * coefs[deg])
            # Downscale (a * x**n) to the float_precision_var range
            target_divided_by_approx = model.NewIntVar(
                lb=lb,
                ub=ub,
                name=f"{coefs[deg]}*{var.Name()}**{deg} / ({float_precision_var**(deg-1)})",
            )
            model.AddDivisionEquality(
                target_divided_by_approx,
                target_times_coef,
                float_precision_var ** (deg - 1),
            )

            polynom_value += target_divided_by_approx

    if verbose:
        print("Polynom", polynom_value)

    polynom_var = model.NewIntVar(
        polynom_value_lb, polynom_value_ub, name=f"{var.Name()}_polynom"
    )
    model.Add(polynom_var == polynom_value)
    return polynom_var


def lookup_value_in_dict(
    model: cp_model.CpModel,
    key_var: cp_model.IntVar,
    mapping: Dict[int, int],
    mapping_name: str = "mapping",
):
    """Creates a new variable equals to mapping[key_var]
    If the value is absent from the mapping, the value_var takes the value 0.
    """
    value_var = model.NewIntVar(
        min(mapping.values()), max(mapping.values()), f"{mapping_name}_{key_var.Name()}"
    )
    for mapping_key, mapping_value in mapping.items():
        key_var_is_equal_to = create_boolean_is_equal_to(model, key_var, mapping_key)
        model.Add(value_var == mapping_value).OnlyEnforceIf(key_var_is_equal_to)

    return value_var


# TODO : Make this work with variable bounds


def exp_of_x(
    model: cp_model.CpModel,
    var: cp_model.IntVar,
    float_precision_var=FLOAT_APPROX_PRECISION,
    float_precision_image=FLOAT_APPROX_PRECISION,
):
    lb = var.Proto().domain[0]
    ub = var.Proto().domain[1]
    x_to_exp_x = {
        x: round(exp(x / float_precision_var) * float_precision_image)
        for x in range(lb, ub + 1)
    }
    exp_of_var = lookup_value_in_dict(model, var, x_to_exp_x, mapping_name="exp")
    return exp_of_var


def log_of_x(
    model: cp_model.CpModel,
    var: cp_model.IntVar,
    float_precision_var=FLOAT_APPROX_PRECISION,
    float_precision_image=FLOAT_APPROX_PRECISION,
):
    # Log is only defined for x > 0
    lb = max(var.Proto().domain[0], 1)
    ub = max(var.Proto().domain[1], 1)
    assert lb <= ub
    x_to_log_x = {
        x: round(log(x / float_precision_var) * float_precision_image)
        for x in range(lb, ub + 1)
    }
    log_of_var = lookup_value_in_dict(model, var, x_to_log_x, mapping_name="log")
    return log_of_var
