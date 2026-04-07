import main_pipeline

FILENAME = 'training_data.csv'


def main():
    task_choice = input("Train pipeline, classifier, or regressor? (training will overwrite current save) (p / c / r) ")
    debug = input("Debug mode? (y / n) ")
    graph = input("Graph mode? (y / n) ")

    debug_enabled = debug.lower() == "y"
    graph_enabled = graph.lower() == "y"

    if task_choice.lower() == "p":
        main_pipeline.main(FILENAME, 'classifier', debug_enabled, graph_enabled)
        main_pipeline.main(FILENAME, 'regressor', debug_enabled, graph_enabled)
    elif task_choice.lower() == "c":
        main_pipeline.main(FILENAME, 'classifier', debug_enabled, graph_enabled)
    elif task_choice.lower() == "r":
        main_pipeline.main(FILENAME, 'regressor', debug_enabled, graph_enabled)


if __name__ == "__main__":
    main()
