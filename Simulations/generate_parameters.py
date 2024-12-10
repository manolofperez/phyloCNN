#!/usr/bin/env python3

import numpy as np
from pyDOE import lhs

def get_user_input():
    """
    Prompts the user to select a model, input parameter bounds, number of simulations,
    and output file name, using default values if no input is provided.

    Returns
    -------
    model : str
        Selected model name.
    bounds : numpy.ndarray
        Array of shape (p, 2) containing the lower and upper bounds for each parameter.
    param_names : list
        List of parameter names.
    n_samples : int
        Number of samples to generate.
    output_file : str
        Name of the output file.
    """
    print("Select the model for which you want to generate parameters:")
    print("Options: BD, BDEI, BDSS, BiSSE")
    model = input("Enter the model name [default: BDSS]: ").strip().upper()
    if not model:
        model = "BDSS"
        print(f"No model provided. Using default: {model}")
    elif model not in ["BD", "BDEI", "BDSS", "BISSE"]:
        print(f"Invalid model '{model}'. Using default: BDSS")
        model = "BDSS"

    # Define default bounds and parameter names for each model
    default_params = {
        "BD": {
            "param_names": ['R_nought', 'infectious_period', 'tree_size', 'sampling_proba'],
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
            "param_names": ['R_nought', 'incubation_time', 'infectious_time', 'tree_size', 'sampling_proba'],
            "default_bounds": {
                'R_nought': [1, 5],
                'incubation_time': [0.2, 50],
                'infectious_time': [1, 10],
                'tree_size': [200, 500],
                'sampling_proba': [0.01, 1]
            },
            "default_output_file": "parameters_BDEI.txt",
            "default_n_samples": 10000
        },
        "BDSS": {
            "param_names": ['R_nought', 'x_transmission', 'fraction_1', 'infectious_period', 'tree_size', 'sampling_proba'],
            "default_bounds": {
                'R_nought': [1, 5],
                'x_transmission': [3, 10],
                'fraction_1': [0.05, 0.2],
                'infectious_period': [1, 10],
                'tree_size': [200, 500],
                'sampling_proba': [0.01, 1]
            },
            "default_output_file": "parameters_BDSS.txt",
            "default_n_samples": 10000
        },
        "BISSE": {
            "param_names": ['lambda1', 'turnover', 'sampling_frac', 'tree_size', 'lambda2_ratio', 'q01_ratio'],
            "default_bounds": {
                'lambda1': [0.01, 1.0],
                'turnover': [0.0, 1.0],
                'sampling_frac': [0.01, 1.0],
                'tree_size': [200, 500],
                'lambda2_ratio': [0.1, 1.0],
                'q01_ratio': [0.01, 0.1]
            },
            "default_output_file": "parameters_BiSSE.txt",
            "default_n_samples": 10000
        }
    }

    # Get the parameters for the selected model
    model_params = default_params[model]
    param_names = model_params["param_names"]
    default_bounds = model_params["default_bounds"]
    default_output_file = model_params["default_output_file"]
    default_n_samples = model_params["default_n_samples"]

    print("\nEnter the parameter bounds for each parameter.")
    print("For each parameter, provide the lower and upper bounds separated by a comma.")
    print("Press Enter to use the default value shown in brackets.")
    print("Parameters are:")

    bounds_list = []
    for param in param_names:
        default_lower, default_upper = default_bounds[param]
        while True:
            try:
                bound_input = input(f"Enter bounds for {param} [default: {default_lower},{default_upper}]: ").strip()
                if not bound_input:
                    lower, upper = default_lower, default_upper
                else:
                    lower, upper = map(float, bound_input.split(','))
                    if lower >= upper:
                        print("Error: Lower bound must be less than upper bound.")
                        continue
                bounds_list.append([lower, upper])
                break
            except ValueError:
                print("Invalid input. Please enter two numbers separated by a comma.")

    bounds = np.array(bounds_list)

    # Get the number of samples
    while True:
        try:
            n_samples_input = input(f"Enter the number of simulations to generate [default: {default_n_samples}]: ").strip()
            if not n_samples_input:
                n_samples = default_n_samples
            else:
                n_samples = int(n_samples_input)
                if n_samples <= 0:
                    print("Error: Number of simulations must be a positive integer.")
                    continue
            break
        except ValueError:
            print("Invalid input. Please enter a positive integer.")

    # Get the output file name
    output_file = input(f"Enter the output file name [default: {default_output_file}]: ").strip()
    if not output_file:
        output_file = default_output_file
        print(f"No file name provided. Using default: {output_file}")

    return model, bounds, param_names, n_samples, output_file

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
    # Get user inputs
    model, bounds, param_names, n_samples, output_file = get_user_input()

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
        index = np.arange(len(params))
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
        incubation_time = params[:, 1]
        infectious_time = params[:, 2]
        tree_size = params[:, 3].astype(int)
        sampling_proba = params[:, 4]

        # Compute intermediate values
        removal_rate = 1 / infectious_time
        incubation_rate = 1 / incubation_time
        incubation_ratio = incubation_rate / removal_rate
        transmission_rate = R_nought * removal_rate

        # Prepare the final parameter array
        index = np.arange(len(params))
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
        x_transmission = params[:, 1]
        fraction_1 = params[:, 2]
        infectious_period = params[:, 3]
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
        index = np.arange(len(params))

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

    elif model == "BISSE":
        # Extract parameters
        lambda1 = params[:, 0]
        turnover = params[:, 1]
        sampling_frac = params[:, 2]
        tree_size = params[:, 3].astype(int)
        lambda2_ratio = params[:, 4]
        q01_ratio = params[:, 5]

        # Compute intermediate values
        lambda2 = lambda1 * lambda2_ratio
        mu1 = lambda1 * turnover
        mu2 = lambda2 * turnover
        net_rate1 = lambda1 - mu1
        net_rate2 = lambda2 - mu2
        q01 = lambda1 * q01_ratio
        q10 = lambda1 * q01_ratio  # Assuming q10_ratio is the same as q01_ratio

        # Prepare the final parameter array
        index = np.arange(len(params))

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

        # Define header for the output file (header will not be used)
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

        # Save the final parameters to the specified output file WITHOUT header
        np.savetxt(
            output_file,
            params_final,
            delimiter='\t',
            comments='',   # Ensure no comments are added
            fmt='%f'
        )

    else:
        print("Invalid model selected.")
        return

    print(f"\nParameter samples for model '{model}' have been saved to {output_file}")

if __name__ == "__main__":
    main()
