from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound, LogExpectedImprovement
from botorch.optim import optimize_acqf
from baselines.GP import GP_Wrapper, GP_MAP_Wrapper, Vanilla_GP_Wrapper
from data import *
import time
from infras.randutils import *
from typing import Dict, List

import pandas as pd
from pathlib import Path
import torch
import time

def _extract_hyperparams(model) -> Dict[str, float]:
    """Extracts hyperparameters from a trained GP model"""
    params = {}
    
    #Mean
    if hasattr(model.gp_model.mean_module, 'constant'):
        params['mean_constant'] = float(model.gp_model.mean_module.constant.item())

    # Get kernel parameters
    if hasattr(model.gp_model.covar_module, 'base_kernel'):
        kernel = model.gp_model.covar_module.base_kernel
    else:
        kernel = model.gp_model.covar_module
        
    # Lengthscales (ARD or single)
    if hasattr(kernel, 'lengthscale'):
        ls = kernel.lengthscale.detach().cpu().numpy()
        if ls.shape[1] > 1:  # ARD case
            for i in range(ls.shape[1]):
                params[f'lengthscale_dim_{i}'] = float(ls[0, i])
        else:
            params['lengthscale'] = float(ls[0, 0])
    
    # Output scale
    if hasattr(model.gp_model.covar_module, 'outputscale'):
        params['outputscale'] = float(model.gp_model.covar_module.outputscale.detach().cpu().numpy())
    
    # Noise variance
    if hasattr(model.gp_model.likelihood, 'noise'):
        params['noise'] = float(model.gp_model.likelihood.noise.detach().cpu().numpy())
    
    # Kernel-specific parameters
    if hasattr(kernel, 'nu'):  # Matern nu
        params['nu'] = float(kernel.nu)
    elif hasattr(kernel, 'alpha'):  # GeneralCauchy alpha
        params['alpha'] = float(kernel.alpha)
        params['beta'] = float(kernel.beta)
    
    return params


def BO_loop_GP(func_name, dataset, seed, num_step=200, beta=1.5, if_ard=False, if_softplus=True, acqf_type="UCB", if_matern=True, set_ls=False, device="cpu"):
    best_y = []
    time_list = []
    dim = dataset.func.dims

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "hyperparams" / func_name / "bo_gp"
    output_dir.mkdir(parents=True, exist_ok=True)

    hyperparam_history = []

    for i in range(1, num_step+1):
        start_time = time.time()
        X, Y = dataset.get_data(normalize=True)
        X, Y = X.to(device), Y.to(device)
        best_y_before = dataset.get_curr_max_unnormed()
        model = GP_Wrapper(X, Y, if_ard, if_softplus, if_matern=if_matern, set_ls=set_ls)

        if func_name in ["Ackley150"]:
            model.train_model(1000, 0.01)
        elif func_name in ["Ackley"]:
            model.train_model(None, None, optim="botorch")
        elif func_name == "Hartmann6":
            model.train_model(400, 0.01, optim="RMSPROP")
        else:
            model.train_model(500, 0.1)

        # Extract hyperparameters after training
        hyperparams = _extract_hyperparams(model)
        hyperparams['iteration'] = i
        hyperparam_history.append(hyperparams)

        if acqf_type == "UCB":
            acqf = UpperConfidenceBound(model=model.gp_model, beta=beta, maximize=True).to(device)
        elif acqf_type == "EI":
            acqf = ExpectedImprovement(model=model.gp_model, best_f=Y.max()).to(device)
        elif acqf_type == "LogEI":
            acqf = LogExpectedImprovement(model=model.gp_model, best_f=Y.max()).to(device)
        else:
            raise NotImplementedError

        try:
            new_x, _ = optimize_acqf(
                acq_function=acqf,
                bounds=torch.tensor([[0.0] * dim, [1.0] * dim], device=device),
                q=1,
                num_restarts=10,
                raw_samples=1000,
                options={},
            )
        except Exception as e:
            print(f"ERROR during opt acqf ({e}), using random point")
            new_x = torch.rand(dim, device=device)

        dataset.add(new_x.detach().cpu())
        time_used = time.time() - start_time
        time_list.append(time_used)
        best_y_after = dataset.get_curr_max_unnormed()
        itr = dataset.X.shape[0]
        print(f"Seed: {seed} --- At itr: {itr}: best before={best_y_before}, best after={best_y_after}, current query: {dataset.y[-1]}, time={time_used:.3f}s", flush=True)
        best_y.append(best_y_before)

        # Save hyperparameters to CSV every iteration (safe checkpointing)
        df_hyperparams = pd.DataFrame(hyperparam_history)
        output_path = output_dir / f"hyperparams_{func_name}_bo_gp_seed{seed}.csv"
        df_hyperparams.to_csv(output_path, index=False)

    return best_y, time_list


def Vanilla_BO_loop(func_name, dataset, seed, num_step=200, device="cpu"):
    best_y = []
    time_list = []
    dim = dataset.func.dims

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "hyperparams" / func_name / "vanilla"
    output_dir.mkdir(parents=True, exist_ok=True)

    hyperparam_history = []

    for i in range(1, num_step + 1):
        start_time = time.time()
        X, Y = dataset.get_data(normalize=True)
        X, Y = X.to(device), Y.to(device)
        best_y_before = dataset.get_curr_max_unnormed()
        model = Vanilla_GP_Wrapper(X, Y)
        model.train_model()

        # Extract hyperparameters after training
        hyperparams = _extract_hyperparams(model)
        hyperparams['iteration'] = i
        hyperparam_history.append(hyperparams)

        ls = model.gp_model.covar_module.base_kernel.lengthscale
        print(f"ls mean: {ls.mean()}, ls std: {ls.std()}, max: {ls.max()}, min: {ls.min()}")

        acqf = LogExpectedImprovement(model=model.gp_model, best_f=Y.max())
        new_x, _ = optimize_acqf(
            acq_function=acqf,
            bounds=torch.tensor([[0.0] * dim, [1.0] * dim], device=device),
            q=1,
            num_restarts=10,
            raw_samples=1000,
            options={},
        )
        dataset.add(new_x.detach().cpu())
        time_used = time.time() - start_time
        time_list.append(time_used)
        best_y_after = dataset.get_curr_max_unnormed()

        print(f"Seed: {seed} --- At itr: {i}: best before={best_y_before}, best after={best_y_after}, current query: {dataset.y[-1]}", flush=True)
        best_y.append(best_y_before)

        # Save hyperparameters every iteration
        df_hyperparams = pd.DataFrame(hyperparam_history)
        output_path = output_dir / f"hyperparams_{func_name}_vanilla_seed{seed}.csv"
        df_hyperparams.to_csv(output_path, index=False)

    return best_y, time_list


def BO_loop_GP_MAP(func_name, dataset, seed, num_step=200, beta=1.5, if_ard=True, optim_type="LBFGS", acqf_type="UCB", ls_prior_type="Gamma", set_ls=False, if_matern=False, device="cpu"):
    best_y = []
    time_list = []
    dim = dataset.func.dims

    base_dir = Path(__file__).parent.parent
    kernel_name = "matern" if if_matern else "rbf"
    output_dir = base_dir / "hyperparams" / func_name / f"map_{kernel_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    hyperparam_history = []

    for i in range(1, num_step + 1):
        start_time = time.time()
        X, Y = dataset.get_data(normalize=True)
        X, Y = X.to(device), Y.to(device)
        best_y_before = dataset.get_curr_max_unnormed()
        model = GP_MAP_Wrapper(X, Y, if_ard=if_ard, if_matern=if_matern, optim_type=optim_type,
                               ls_prior_type=ls_prior_type, device=device, set_ls=set_ls)
        model.train_model()

        # Extract hyperparameters after training
        hyperparams = _extract_hyperparams(model)
        hyperparams['iteration'] = i
        hyperparam_history.append(hyperparams)

        if acqf_type == "UCB":
            acqf = UpperConfidenceBound(model=model.gp_model, beta=beta, maximize=True).to(device)
        elif acqf_type == "EI":
            acqf = ExpectedImprovement(model=model.gp_model, best_f=Y.max()).to(device)
        else:
            raise NotImplementedError

        new_x, _ = optimize_acqf(
            acq_function=acqf,
            bounds=torch.tensor([[0.0] * dim, [1.0] * dim], device=device),
            q=1,
            num_restarts=10,
            raw_samples=1000,
            options={},
        )
        dataset.add(new_x.detach().cpu())
        time_used = time.time() - start_time
        time_list.append(time_used)
        best_y_after = dataset.get_curr_max_unnormed()

        print(f"Seed: {seed} --- At itr: {i}: best before={best_y_before}, best after={best_y_after}, current query: {dataset.y[-1]}", flush=True)
        best_y.append(best_y_before)

        # Save hyperparameters every iteration
        df_hyperparams = pd.DataFrame(hyperparam_history)
        output_path = output_dir / f"hyperparams_{func_name}_map_{kernel_name}_seed{seed}.csv"
        df_hyperparams.to_csv(output_path, index=False)

    return best_y, time_list
