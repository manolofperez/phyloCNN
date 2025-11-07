#!/usr/bin/env python3

import numpy as np
from pyDOE import lhs
import argparse
import sys

def parse_bounds(bound_str, param_name):
    """
    Parses a string containing lower and upper bounds separated by a comma.
    
    Parameters
    ----------
    bound_str : str
        String of the form 'lower,upper'.
    param_name : str
        Name of the parameter (for error messages).
    
    Returns
    -------
    lower : float
        Lower bound.
    upper : float
        Upper bound.
    """
    try:
        lower_str, upper_str = bound_str.split(',')
        lower = float(lower_str)
        upper = float(upper_str)
        if lower >= upper:
            raise ValueError(f"Lower bound must be less than upper bound for parameter '{param_name}'.")
        return lower, upper
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid bounds for parameter '{param_name}': {e}")

def generate_parameter_samples(n_samples, bounds):
    """
    Generates Latin Hypercube samples within specified bounds.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    bounds : numpy.ndarray
        Array of shape (p, 2) where p is the number of parameters.
        Each row contains the lower and upper bounds for a parameter.

    Returns
    -------
    samples : numpy.ndarray
        Array of shape (n_samples, p) containing the parameter samples.
    """
    p = bounds.shape[0]
    # Generate Latin Hypercube samples
    lh_samples = lhs(p, samples=n_samples)
    # Scale samples to the specified bounds
    samples = lh_samples * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    return samples

def main():
    # Define default bounds and parameter names for each model
    default_params = {
        "BD": {
            "param_names": ['R_nought', 'infectious_period', 'tree_size', 'sampling_proba'],
            "short_names": {'R_nought': 'r', 'infectious_period': 'i', 'tree_size': 's', 'sampling_proba': 'p'},
            "default_bounds": {
                'R_nought': [1, 5],
                'infectious_period': [1, 10],
                'tree_size': [200, 500],
                'sampling_proba': [0.01, 1]
            },
            "default_output_file": "parameters_BD.txt",
            "default_n_samples": 10000
        },
        "BDEI": {
            "param_names": ['R_nought', 'infectious_time', 'incubation_factor', 'tree_size', 'sampling_proba'],
            "short_names": {'R_nought': 'r', 'infectious_time': 'i', 'incubation_factor': 'e', 'tree_size': 's', 'sampling_proba': 'p'},
            "default_bounds": {
                'R_nought': [1, 5],
                'infectious_time': [1, 10],
                'incubation_factor': [0.2, 5],
                'tree_size': [200, 500],
                'sampling_proba': [0.01, 1]
            },
            "default_output_file": "parameters_BDEI.txt",
            "default_n_samples": 10000
        },
        "BDSS": {
            "param_names": ['R_nought', 'infectious_period', 'x_transmission', 'fraction_1', 'tree_size', 'sampling_proba'],
            "short_names": {'R_nought': 'r', 'infectious_period': 'i', 'x_transmission': 'x', 'fraction_1': 'f', 'tree_size': 's', 'sampling_proba': 'p'},
            "default_bounds": {
                'R_nought': [1, 5],
                'infectious_period': [1, 10],
                'x_transmission': [3, 10],
                'fraction_1': [0.05, 0.2],
                'tree_size': [200, 500],
                'sampling_proba': [0.01, 1]
            },
            "default_output_file": "parameters_BDSS.txt",
            "default_n_samples": 10000
        },
        "BD_div": {
            "param_names": ['turnover_rate', 'birth_rate', 'sampling_frac', 'tree_size'],
            "short_names": {'turnover_rate': 't', 'birth_rate': 'l', 'sampling_frac': 'p', 'tree_size': 's'},
            "default_bounds": {
                'turnover_rate': [0.01, 1],
                'birth_rate': [0.01, 0.5],
                'sampling_frac': [0.01, 1],
                'tree_size': [200, 500]

            },
            "default_output_file": "parameters_BD_div.txt",
            "default_n_samples": 10000
        },
        "BISSE": {
            "param_names": ['lambda1', 'turnover', 'lambda2_ratio', 'q01_ratio', 'tree_size', 'sampling_frac'],
            "short_names": {'lambda1': 'l0', 'turnover': 't', 'lambda2_ratio': 'l1', 'q01_ratio': 'q', 'tree_size': 's', 'sampling_frac': 'p'},
            "default_bounds": {
                'lambda1': [0.01, 1.0],
                'turnover': [0.0, 1.0],
                'lambda2_ratio': [0.1, 1.0],
                'q01_ratio': [0.01, 0.1],
                'tree_size': [200, 500],
                'sampling_frac': [0.01, 1.0]
            },
            "default_output_file": "parameters_BiSSE.txt",
            "default_n_samples": 10000
        }
    }

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate parameter samples for various models.')
    parser.add_argument('-m', '--model', type=str, choices=['BD', 'BDEI', 'BDSS', 'BD_div', 'BISSE'], default='BDSS',
                        help='Model name (default: BDSS)')
    parser.add_argument('-n', '--n_samples', type=int, help='Number of samples to generate')
    parser.add_argument('-o', '--output', type=str, help='Output file name')

    # Parse known arguments to get the model
    args, remaining_args = parser.parse_known_args()
    model = args.model.upper()

    # Get the parameters for the selected model
    if model not in default_params:
        print(f"Invalid model '{model}'. Available models are: BD, BDEI, BDSS, BISSE")
        sys.exit(1)

    model_params = default_params[model]
    param_names = model_params["param_names"]
    short_names = model_params["short_names"]
    default_bounds = model_params["default_bounds"]
    default_output_file = model_params["default_output_file"]
    default_n_samples = model_params["default_n_samples"]

    # Set n_samples
    n_samples = args.n_samples if args.n_samples is not None else default_n_samples

    # Set output file
    output_file = args.output if args.output is not None else default_output_file

    # Add arguments for parameter bounds using short names
    for param in param_names:
        short_name = short_names[param]
        parser.add_argument(f'-{short_name}', type=str, help=f'Bounds for {param} (format: lower,upper)')

    # Re-parse arguments including parameter-specific bounds
    args = parser.parse_args()

    # Build bounds array
    bounds_list = []
    for param in param_names:
        short_name = short_names[param]
        bound_str = getattr(args, short_name, None)
        default_lower, default_upper = default_bounds[param]
        if bound_str is not None:
            try:
                lower, upper = parse_bounds(bound_str, param)
            except argparse.ArgumentTypeError as e:
                parser.error(str(e))
        else:
            lower, upper = default_lower, default_upper
        bounds_list.append([lower, upper])

    bounds = np.array(bounds_list)

    # Generate parameter samples
    params = generate_parameter_samples(n_samples, bounds)

    # Prepare the final parameter array and header based on the model
    if model == "BD":
        # Extract parameters
        R_nought = params[:, 0]
        infectious_period = params[:, 1]
        tree_size = params[:, 2].astype(int)
        sampling_proba = params[:, 3]

        # Compute intermediate values
        transmission_rate = R_nought / infectious_period
        removal_rate = 1 / infectious_period

        # Prepare the final parameter array
        index = np.arange(n_samples)
        params_final = np.column_stack((
            index,
            R_nought,
            transmission_rate,
            removal_rate,
            sampling_proba,
            infectious_period,
            tree_size
        ))

        # Define header for the output file
        header = '\t'.join([
            'index',
            'R_nought',
            'transmission_rate',
            'removal_rate',
            'sampling_proba',
            'infectious_time',
            'tree_size'
        ])

        # Save the final parameters to the specified output file
        np.savetxt(
            output_file,
            params_final,
            header=header,
            delimiter='\t',
            comments='',
            fmt='%f'
        )

    elif model == "BDEI":
        # Extract parameters
        R_nought = params[:, 0]
        infectious_time = params[:, 1]
        incubation_factor = params[:, 2]
        tree_size = params[:, 3].astype(int)
        sampling_proba = params[:, 4]

        # Compute intermediate values
        removal_rate = 1 / infectious_time
        incubation_rate = 1 / (incubation_factor * removal_rate)
        incubation_ratio = incubation_rate / removal_rate
        transmission_rate = R_nought * removal_rate

        # Prepare the final parameter array
        index = np.arange(n_samples)
        params_final = np.column_stack((
            index,
            R_nought,
            transmission_rate,
            removal_rate,
            sampling_proba,
            incubation_ratio,
            incubation_rate,
            infectious_time,
            tree_size
        ))

        # Define header for the output file
        header = '\t'.join([
            'index',
            'R_nought',
            'transmission_rate',
            'removal_rate',
            'sampling_proba',
            'incubation_ratio',
            'incubation_rate',
            'infectious_time',
            'tree_size'
        ])

        # Save the final parameters to the specified output file
        np.savetxt(
            output_file,
            params_final,
            header=header,
            delimiter='\t',
            comments='',
            fmt='%f'
        )

    elif model == "BDSS":
        # Extract parameters
        R_nought = params[:, 0]
        infectious_period = params[:, 1]
        x_transmission = params[:, 2]
        fraction_1 = params[:, 3]
        tree_size = params[:, 4].astype(int)
        sampling_proba = params[:, 5]

        # Compute intermediate values
        fx = fraction_1 * x_transmission
        la = R_nought / infectious_period
        Bss = la * fx / (fx + 1 - fraction_1)
        Bnn = la - Bss

        # Compute additional parameters
        tr_rate_1_1 = Bss
        tr_rate_2_2 = Bnn
        tr_rate_1_2 = Bnn * x_transmission
        tr_rate_2_1 = Bss / x_transmission
        removal_rate = 1 / infectious_period

        # Prepare the final parameter array
        index = np.arange(n_samples)

        params_final = np.column_stack((
            index,
            R_nought,
            tr_rate_1_1,
            tr_rate_2_2,
            tr_rate_1_2,
            tr_rate_2_1,
            removal_rate,
            sampling_proba,
            R_nought,            # R_nought_1
            R_nought,            # R_nought_2
            R_nought,            # R_nought_verif
            tree_size,
            x_transmission,
            fraction_1,
            infectious_period
        ))

        # Define header for the output file
        header = '\t'.join([
            'index',
            'R_nought',
            'tr_rate_1_1',
            'tr_rate_2_2',
            'tr_rate_1_2',
            'tr_rate_2_1',
            'removal_rate',
            'sampling_proba',
            'R_nought_1',
            'R_nought_2',
            'R_nought_verif',
            'tree_size',
            'x_transmission',
            'fraction_1',
            'infectious_period'
        ])

        # Save the final parameters to the specified output file
        np.savetxt(
            output_file,
            params_final,
            header=header,
            delimiter='\t',
            comments='',
            fmt='%f'
        )

    elif model == "BD_div":
        # Extract parameters
        turnover_rate = params[:, 0]
        birth_rate = params[:, 1]
        sampling_frac = params[:, 2]
        tree_size = params[:, 3].astype(int)

        # Compute intermediate values
        extinction_rate = birth_rate * turnover_rate

        # Prepare the final parameter array
        index = np.arange(len(params))
        params_final = np.column_stack((
            index,
            turnover_rate,
            birth_rate,
            extinction_rate,
            sampling_frac,
            tree_size
        ))

        # Define header for the output file
        header = '\t'.join([
            'index',
            'turnover_rate',
            'birth_rate',
            'extinction_rate',
            'sampling_frac',
            'tree_size'
        ])

        # Save the final parameters to the specified output file
        np.savetxt(
            output_file,
            params_final,
            header=header,
            delimiter='\t',
            comments='',
            fmt='%f'
        )

    elif model == "BISSE":
        # Extract parameters
        lambda1 = params[:, 0]
        turnover = params[:, 1]
        lambda2_ratio = params[:, 2]
        q01_ratio = params[:, 3]
        tree_size = params[:, 4].astype(int)
        sampling_frac = params[:, 5]

        # Compute intermediate values
        lambda2 = lambda1 * lambda2_ratio
        mu1 = lambda1 * turnover
        mu2 = lambda2 * turnover
        net_rate1 = lambda1 - mu1
        net_rate2 = lambda2 - mu2
        q01 = lambda1 * q01_ratio
        q10 = lambda1 * q01_ratio  # Assuming q10_ratio is the same as q01_ratio

        # Prepare the final parameter array
        index = np.arange(n_samples)

        params_final = np.column_stack((
            index,
            lambda1,
            lambda2,
            turnover,
            sampling_frac,
            tree_size,
            mu1,
            mu2,
            net_rate1,
            net_rate2,
            q01,
            q10,
            lambda2_ratio,
            q01_ratio
        ))

        # Define header for the output file
        header = '\t'.join([
            'index',
            'lambda1',
            'lambda2',
            'turnover',
            'sampling_frac',
            'tree_size',
            'mu1',
            'mu2',
            'net_rate1',
            'net_rate2',
            'q01',
            'q10',
            'lambda2_ratio',
            'q01_ratio'
        ])

        # Save the final parameters to the specified output file
        np.savetxt(
            output_file,
            params_final,
            delimiter='\t',
            comments='',
            fmt='%f'
        )

    else:
        print("Invalid model selected.")
        sys.exit(1)

    print(f"\nParameter samples for model '{model}' have been saved to {output_file}")

if __name__ == "__main__":
    main()
