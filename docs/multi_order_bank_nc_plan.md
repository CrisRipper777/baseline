# Multi-Order Bank NC benchmark plan

## Frozen scope

The future formal plan contains 15 dataset/readout jobs: five NC datasets (Movies, Toys, Grocery, ele-fashion, Reddit-S) crossed with terminal, uniform, and gpr. Each job uses one command with seed=42 and num_runs=3. The task runner creates run seeds 42, 43, and 44 internally. Do not split seeds into separate shell commands.

Every job writes its three per-run checkpoints and Hydra result files under a dataset/readout-specific directory beneath outputs/multi_order_bank_nc/. Checkpoint paths become best_run1.pt, best_run2.pt, and best_run3.pt.

## Exact future commands

The commands below are prepared for a future formal launch. They have not been executed.

    PYTHONPATH=src conda run --no-capture-output -n yhf_env python -m src.main dataset=Movies task=nc model=multi_order_bank model.readout=terminal seed=42 num_runs=3 device=cuda:0 task.save_ckpt_path=outputs/multi_order_bank_nc/Movies/terminal/best.pt hydra.run.dir=outputs/multi_order_bank_nc/Movies/terminal/run
    PYTHONPATH=src conda run --no-capture-output -n yhf_env python -m src.main dataset=Movies task=nc model=multi_order_bank model.readout=uniform seed=42 num_runs=3 device=cuda:0 task.save_ckpt_path=outputs/multi_order_bank_nc/Movies/uniform/best.pt hydra.run.dir=outputs/multi_order_bank_nc/Movies/uniform/run
    PYTHONPATH=src conda run --no-capture-output -n yhf_env python -m src.main dataset=Movies task=nc model=multi_order_bank model.readout=gpr seed=42 num_runs=3 device=cuda:0 task.save_ckpt_path=outputs/multi_order_bank_nc/Movies/gpr/best.pt hydra.run.dir=outputs/multi_order_bank_nc/Movies/gpr/run
    PYTHONPATH=src conda run --no-capture-output -n yhf_env python -m src.main dataset=Toys task=nc model=multi_order_bank model.readout=terminal seed=42 num_runs=3 device=cuda:0 task.save_ckpt_path=outputs/multi_order_bank_nc/Toys/terminal/best.pt hydra.run.dir=outputs/multi_order_bank_nc/Toys/terminal/run
    PYTHONPATH=src conda run --no-capture-output -n yhf_env python -m src.main dataset=Toys task=nc model=multi_order_bank model.readout=uniform seed=42 num_runs=3 device=cuda:0 task.save_ckpt_path=outputs/multi_order_bank_nc/Toys/uniform/best.pt hydra.run.dir=outputs/multi_order_bank_nc/Toys/uniform/run
    PYTHONPATH=src conda run --no-capture-output -n yhf_env python -m src.main dataset=Toys task=nc model=multi_order_bank model.readout=gpr seed=42 num_runs=3 device=cuda:0 task.save_ckpt_path=outputs/multi_order_bank_nc/Toys/gpr/best.pt hydra.run.dir=outputs/multi_order_bank_nc/Toys/gpr/run
    PYTHONPATH=src conda run --no-capture-output -n yhf_env python -m src.main dataset=Grocery task=nc model=multi_order_bank model.readout=terminal seed=42 num_runs=3 device=cuda:0 task.save_ckpt_path=outputs/multi_order_bank_nc/Grocery/terminal/best.pt hydra.run.dir=outputs/multi_order_bank_nc/Grocery/terminal/run
    PYTHONPATH=src conda run --no-capture-output -n yhf_env python -m src.main dataset=Grocery task=nc model=multi_order_bank model.readout=uniform seed=42 num_runs=3 device=cuda:0 task.save_ckpt_path=outputs/multi_order_bank_nc/Grocery/uniform/best.pt hydra.run.dir=outputs/multi_order_bank_nc/Grocery/uniform/run
    PYTHONPATH=src conda run --no-capture-output -n yhf_env python -m src.main dataset=Grocery task=nc model=multi_order_bank model.readout=gpr seed=42 num_runs=3 device=cuda:0 task.save_ckpt_path=outputs/multi_order_bank_nc/Grocery/gpr/best.pt hydra.run.dir=outputs/multi_order_bank_nc/Grocery/gpr/run
    PYTHONPATH=src conda run --no-capture-output -n yhf_env python -m src.main dataset=ele-fashion task=nc model=multi_order_bank model.readout=terminal seed=42 num_runs=3 device=cuda:0 task.save_ckpt_path=outputs/multi_order_bank_nc/ele-fashion/terminal/best.pt hydra.run.dir=outputs/multi_order_bank_nc/ele-fashion/terminal/run
    PYTHONPATH=src conda run --no-capture-output -n yhf_env python -m src.main dataset=ele-fashion task=nc model=multi_order_bank model.readout=uniform seed=42 num_runs=3 device=cuda:0 task.save_ckpt_path=outputs/multi_order_bank_nc/ele-fashion/uniform/best.pt hydra.run.dir=outputs/multi_order_bank_nc/ele-fashion/uniform/run
    PYTHONPATH=src conda run --no-capture-output -n yhf_env python -m src.main dataset=ele-fashion task=nc model=multi_order_bank model.readout=gpr seed=42 num_runs=3 device=cuda:0 task.save_ckpt_path=outputs/multi_order_bank_nc/ele-fashion/gpr/best.pt hydra.run.dir=outputs/multi_order_bank_nc/ele-fashion/gpr/run
    PYTHONPATH=src conda run --no-capture-output -n yhf_env python -m src.main dataset=Reddit-S task=nc model=multi_order_bank model.readout=terminal seed=42 num_runs=3 device=cuda:0 task.save_ckpt_path=outputs/multi_order_bank_nc/Reddit-S/terminal/best.pt hydra.run.dir=outputs/multi_order_bank_nc/Reddit-S/terminal/run
    PYTHONPATH=src conda run --no-capture-output -n yhf_env python -m src.main dataset=Reddit-S task=nc model=multi_order_bank model.readout=uniform seed=42 num_runs=3 device=cuda:0 task.save_ckpt_path=outputs/multi_order_bank_nc/Reddit-S/uniform/best.pt hydra.run.dir=outputs/multi_order_bank_nc/Reddit-S/uniform/run
    PYTHONPATH=src conda run --no-capture-output -n yhf_env python -m src.main dataset=Reddit-S task=nc model=multi_order_bank model.readout=gpr seed=42 num_runs=3 device=cuda:0 task.save_ckpt_path=outputs/multi_order_bank_nc/Reddit-S/gpr/best.pt hydra.run.dir=outputs/multi_order_bank_nc/Reddit-S/gpr/run

## Launcher and analyzer

Run all 15 prepared jobs sequentially with:

    scripts/run_multi_order_bank_nc.sh

After all checkpoints exist, produce per-dataset/readout mean ± population std, paired same-seed differences (uniform - terminal, gpr - terminal, gpr - uniform), and each GPR run's gamma_text/gamma_visual with:

    PYTHONPATH=src conda run --no-capture-output -n yhf_env python scripts/summarize_multi_order_bank_nc.py --root outputs/multi_order_bank_nc

The analyzer reads run checkpoints and checks expected seeds, best-epoch metadata, validation-accuracy selection, required metrics, and GPR coefficients before writing summary.json and summary.md.

## Smoke record and launch status

The three requested Movies NC smokes completed with seed=42, num_runs=1, and epochs=2. The three imported model NC smokes and their tiny sports-LP interface smokes also completed as documented in imported_baseline_notes.md.

No formal 5-dataset benchmark has started. The launcher and 15 commands are prepared only.
