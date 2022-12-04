import os
import modal
    
LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4","joblib","seaborn","scikit-learn==0.24.2","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("ScalableML_lab1"))
   def f():
       g()

def g():
    import pandas as pd
    import numpy as np
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login(api_key_value="CDqcnm3gyfxjyCO8.TZwOClLOwCqDp33vX0P5Q2nsvNNyEhfBMArwNoPjnb9tUSSKq6I8X35HQ5D2tlJ7")
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("titanic_modal", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/titanic_modal.pkl")
    
    feature_view = fs.get_feature_view(name="titanic_modal", version=1)
    batch_data = feature_view.get_batch_data()
    
    y_pred = model.predict(batch_data)
    # print(y_pred) ##
    person_select = np.random.randint(y_pred.size)
    titanic = y_pred[y_pred.size-1]
    titanic = y_pred[person_select]
    titanic_url = "https://raw.githubusercontent.com/Tilosmsh/IL2223_lab1/main/images/" + str(titanic) + ".jpg"
    print("Passenger predicted: " + str(titanic))
    img = Image.open(requests.get(titanic_url, stream=True).raw)
    img.save("./latest_titanic.png")
    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./latest_titanic.png", "Resources/images", overwrite=True)
    
    titanic_fg = fs.get_feature_group(name="titanic_modal", version=1)
    df = titanic_fg.read()
    # print(df["survived"]) ##
    # label = df.iloc[-1]["survived"]
    label = df.iloc[person_select]["survived"]
    # print(label)
    label_url = "https://raw.githubusercontent.com/Tilosmsh/IL2223_lab1/main/images/" + str(int(label)) + ".jpg"
    print("Titanic actual: " + str(label))
    img = Image.open(requests.get(label_url, stream=True).raw)            
    img.save("./actual_titanic.png")
    dataset_api.upload("./actual_titanic.png", "Resources/images", overwrite=True)
    
    monitor_fg = fs.get_or_create_feature_group(name="titanic_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Titanic Survived Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [titanic],
        'label': [label],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])


    df_recent = history_df.tail(5)
    dfi.export(df_recent, './df_recent_titanic.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent_titanic.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our iris_predictions feature group has examples of all 3 iris flowers
    print("Number of different Titanic predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 2:
        results = confusion_matrix(labels, predictions)
    
        df_cm = pd.DataFrame(results, ['True Dead', 'True Survived'],
                             ['Pred Dead', 'Pred Survived'])
    
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix_titanic.png")
        dataset_api.upload("./confusion_matrix_titanic.png", "Resources/images", overwrite=True)
    else:
        print("You need 2 different titanic predictions to create the confusion matrix.") ##
        print("Run the batch inference pipeline more times until you get 2 different titanic predictions")  ##


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

