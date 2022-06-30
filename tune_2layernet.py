from hebbcl.parameters import parser, set_hpo_args
from hebbcl.tuner import HPOTuner, save_tuner_results


if __name__ == "__main__":
    
    configs = [
        "interleaved_vanilla_1ctx",
        "blocked_vanilla_1ctx",
        "interleaved_vanilla_2ctx",
        "blocked_vanilla_2ctx",
        "interleaved_ojactx_1ctx",
        "blocked_ojactx_1ctx",
        "interleaved_ojactx_2ctx",
        "blocked_ojactx_2ctx",
    ]
    
    for cfg in configs:
        print(f"performing HPO for {cfg}")
        args = parser.parse_args()
        args = set_hpo_args(args, whichmodel=cfg)
        # init tuner
        tuner = HPOTuner(
            args,
            time_budget=60 * 30,
            metric="acc",
            dataset="trees",
            filepath="/datasets/",
            working_dir="ray_tune/",
        )

        tuner.tune(n_samples=600, resources_per_trial={"cpu": 1, "gpu": 0})
        save_tuner_results(tuner.results, args, filename="trees_" + cfg)
