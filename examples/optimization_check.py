import optuna

for study_name in ['classifier-restaurant', 'classifier-laptop']:
    study = optuna.load_study(study_name, storage='sqlite:///optimization.db')
    # fig = optuna.visualization.plot_parallel_coordinate(study)
    # fig.show()

    df = study.trials_dataframe()
    complete = df.state == 'COMPLETE'
    df = df.loc[complete]
    df = df.sort_values(by='value', ascending=False)
    df.to_csv(f'{study_name}.csv', index=False)
