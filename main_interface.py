import main_pipeline

FILENAME = 'training_data.csv' 

def main():
    trainorpass = input("Train pipeline, classifier, or regressor? (training will overwrite current save) (p / c / r) ")

    debug = input("debug mode? (y / n) ")
    graph = input("graph mode? (y / n) ")
    if trainorpass.lower() == "p":  
        main_pipeline.main(FILENAME, 'classifier', debug.lower() == "y", graph.lower() == "y")
        main_pipeline.main(FILENAME, 'regressor', debug.lower() == "y", graph.lower() == "y")
    elif trainorpass.lower() == "c":
        main_pipeline.main(FILENAME, 'classifier', debug.lower() == "y", graph.lower() == "y")
    elif trainorpass.lower() == "r":
        main_pipeline.main(FILENAME, 'regressor', debug.lower() == "y", graph.lower() == "y")

# possibly combine classifier and regressor into one func call to avoid reiniting dataset each time
# create pipeline wide graph
# create forward pass
def forward_pass():
    pass

if __name__ == "__main__":
    main()