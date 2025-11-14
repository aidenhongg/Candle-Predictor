import main_pipeline


FILENAME = 'training_data.csv' 

def main():
    trainorpass = input("Classifier: train model or forward pass? (training will overwrite current save) (t / f)")

    debug = input("debug mode? (y / n)")
    graph = input("graph mode? (y / n)")
    if trainorpass.lower() == "t":
        main_pipeline.main(FILENAME, 'classifier', debug.lower() == "y", graph.lower() == "y")
    else:
        # forward_pass()
        pass
    """
    trainorpass = input("Regeression: train model or forward pass? (training will overwrite current save) (t / f)")
    debug = input("debug mode? (y / n)")
    graph = input("graph mode? (y / n)")

    if trainorpass.lower() == "t":
        main_pipeline.main_classifier(FILENAME, debug.lower() == "y", graph.lower() == "y")
    else:
        # forward_pass()
        pass
    """
def forward_pass():
    pass

if __name__ == "__main__":
    main()