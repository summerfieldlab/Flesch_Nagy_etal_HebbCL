from hebbcl.parameters import parser, set_hpo_args
from hebbcl.tuner import HPOTuner, save_tuner_results, validate_tuner_results


if __name__ == "__main__":

    # configs = [
    #     "interleaved_vanilla_1ctx",
    #     "blocked_vanilla_1ctx",
    #     "interleaved_vanilla_2ctx",
    #     "blocked_vanilla_2ctx",
    #     "interleaved_ojactx_1ctx",
    #     "blocked_ojactx_1ctx",
    #     "interleaved_ojactx_2ctx",
    #     "blocked_ojactx_2ctx",
    # ]

    # configs = [
    #     "interleaved_vanilla_1ctx",
    #     "blocked_vanilla_1ctx",
    #     "interleaved_ojaall_1ctx",
    #     "blocked_ojaall_1ctx",
    #     "interleaved_ojactx_1ctx",
    #     "blocked_ojactx_1ctx",
    # ]

    configs = [
        "blocked_ojaall_1ctx",
    ]
    # 1 hidden layer on blobs (short): -----------------------

    # for cfg in configs:
    #     print(f"performing HPO for {cfg}")
    #     args = parser.parse_args()
    #     args.n_episodes = 8
    #     args.n_layers = 1
    #     args.n_hidden = 100
    #     args.n_features = 27
    #     args.ctx_avg = False
    #     args = set_hpo_args(args, whichmodel=cfg)
    #     args.hpo_scheduler = "asha"
    #     args.hpo_searcher = None
    #     # init tuner
    #     tuner = HPOTuner(
    #         args,
    #         time_budget=60 * 120,
    #         metric="acc",
    #         dataset="blobs",
    #         filepath="/datasets/",
    #         working_dir="ray_temp_env/",
    #         log_dir="ray_logs/",
    #     )

    #     tuner.tune(n_samples=3000, resources_per_trial={"cpu": 1, "gpu": 0})
    #     save_tuner_results(tuner.results, args, filename="blobs_asha_8episodes_" + cfg)

    #     # load best config and validate with independent runs
    #     validate_tuner_results(filename="blobs_asha_8episodes_" + cfg)

    # # # 1 hidden layer on blobs (long): -----------------------

    # for cfg in configs:
    #     print(f"performing HPO for {cfg}")
    #     args = parser.parse_args()
    #     args.n_episodes = 200
    #     args.n_layers = 1
    #     args.n_hidden = 100
    #     args.n_features = 27
    #     args.ctx_avg = False
    #     args = set_hpo_args(args, whichmodel=cfg)
    #     args.hpo_scheduler = "asha"
    #     args.hpo_searcher = None
    #     # init tuner
    #     tuner = HPOTuner(
    #         args,
    #         time_budget=60 * 120,
    #         metric="acc",
    #         dataset="blobs",
    #         filepath="/datasets/",
    #         working_dir="ray_temp_env/",
    #         log_dir="ray_logs/",
    #     )

    #     tuner.tune(n_samples=3000, resources_per_trial={"cpu": 1, "gpu": 0})
    #     save_tuner_results(
    #         tuner.results, args, filename="blobs_asha_200episodes_" + cfg
    #     )

    #     # load best config and validate with independent runs
    #     validate_tuner_results(filename="blobs_asha_200episodes_" + cfg)

    # 2 hidden layers on trees: ----------------------

    # for cfg in configs:
    #     print(f"performing HPO for {cfg}")
    #     args = parser.parse_args()
    #     args.n_episodes = 100
    #     args.n_layers = 2
    #     args.n_hidden = 100
    #     args.n_features = 974
    #     args.ctx_avg = False
    #     args = set_hpo_args(args, whichmodel=cfg)
    #     args.hpo_scheduler = "asha"
    #     args.hpo_searcher = None
        # init tuner
    #    tuner = HPOTuner(
    #        args,
    #        time_budget=60 * 120,
    #        metric="acc",
    #        dataset="trees",
    #        filepath="/datasets/",
    #        working_dir="ray_temp_env/",
    #         log_dir="ray_logs/",
    #        filesuffix="_ds18",
    #    )

    #    tuner.tune(n_samples=2000, resources_per_trial={"cpu": 2, "gpu": 0})
    #    save_tuner_results(tuner.results, args, filename="trees_asha_" + cfg)

    #    # validate best results
    #    validate_tuner_results(filename="trees_asha_" + cfg, njobs=20)

    # # 2 hidden layers on trees: ----------------------

    for cfg in configs:
        print(f"performing HPO for {cfg}")
        args = parser.parse_args()
        args.n_episodes = 100
        args.n_layers = 2
        args.n_hidden = 100
        args.n_features = 18 * 18 * 3 + 2
        args.ctx_avg = False
        args = set_hpo_args(args, whichmodel=cfg)
        args.hpo_scheduler = "asha"
        args.hpo_searcher = None
        # init tuner
        tuner = HPOTuner(
            args,
            time_budget=60 * 120,
            metric="acc",
            dataset="trees",
            filepath="/datasets/",
            working_dir="ray_temp_env/",
            log_dir="ray_logs/",
            filesuffix="_ds18",
        )

        tuner.tune(n_samples=5000, resources_per_trial={"cpu": 2, "gpu": 0})
        save_tuner_results(tuner.results, args, filename="trees_asha_" + cfg)

        # validate best results
        validate_tuner_results(filename="trees_asha_" + cfg, datasuffix="_ds18", njobs=20)



    for cfg in configs:
        print(f"performing HPO for {cfg}")
        args = parser.parse_args()
        args.n_episodes = 100
        args.n_layers = 2
        args.n_hidden = 100
        args.n_features = 18 * 18 * 3 + 2
        args.ctx_avg = False
        args = set_hpo_args(args, whichmodel=cfg)
        args.hpo_scheduler = "asha"
        args.hpo_searcher = None
        # init tuner
        tuner = HPOTuner(
            args,
            time_budget=60 * 120,
            metric="loss",
            dataset="trees",
            filepath="/datasets/",
            working_dir="ray_temp_env/",
            log_dir="ray_logs/",
            filesuffix="_ds18",
        )

        tuner.tune(n_samples=5000, resources_per_trial={"cpu": 2, "gpu": 0})
        save_tuner_results(tuner.results, args, filename="trees_asha_" + cfg +"_optimloss")

        # validate best results
        validate_tuner_results(filename="trees_asha_" + cfg +"_optimloss", datasuffix="_ds18", njobs=20)