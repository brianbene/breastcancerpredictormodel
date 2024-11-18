from flask import Flask, request, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('breast_cancer_model.pkl')


# Root route with a form for 30 features
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Collect 30 feature inputs from the form
            input_data = [float(request.form[f'feature{i}']) for i in range(1, 31)]
            input_data = np.array(input_data).reshape(1, -1)

            # Predict using the model
            prediction = model.predict(input_data)
            result = "Malignant" if prediction[0] == 1 else "Benign"
            return render_template_string('''
            <!doctype html>
            <title>Breast Cancer Prediction</title>
            <style>
                body { font-family: Arial, sans-serif; background-color: #fce4ec; color: #880e4f; text-align: center; padding: 20px; }
                h1 { color: #d81b60; margin-bottom: 0; }
                .banner { background-color: #d81b60; color: white; padding: 10px; margin-bottom: 20px; border-radius: 10px; display: flex; align-items: center; justify-content: center; }
                .ribbon { width: 50px; vertical-align: middle; margin: 0 10px; }
                form { background-color: #ffffff; padding: 20px; border-radius: 10px; display: inline-block; text-align: left; }
                .input-group { display: grid; grid-template-columns: repeat(10, 1fr); gap: 10px; }
                input[type="text"] { padding: 10px; border: 1px solid #d81b60; border-radius: 5px; width: 100%; }
                input[type="submit"] { background-color: #d81b60; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin-top: 10px; }
                .result { font-size: 20px; font-weight: bold; margin-top: 20px; color: #880e4f; }
            </style>
            <div class="banner">
                <img src="https://i.etsystatic.com/9202327/r/il/e20a5a/2282030261/il_794xN.2282030261_g0tb.jpg" class="ribbon" alt="Pink Ribbon">
                <span>Ysais and Brian's Model</span>
                <img src="https://i.etsystatic.com/9202327/r/il/e20a5a/2282030261/il_794xN.2282030261_g0tb.jpg" class="ribbon" alt="Pink Ribbon">
            </div>
            <h1>Breast Cancer Prediction</h1>
            <form method="post">
                <div class="input-group">
                    ''' + ''.join(
                [f'<input type="text" name="feature{i}" placeholder="Feature {i}" required>' for i in range(1, 31)]) + '''
                </div>
                <input type="submit" value="Predict">
            </form>
            <div class="result">Prediction: {{ result }}</div>
            ''', result=result)

        except Exception as e:
            return f"An error occurred: {e}"

    # Render the initial form page
    return '''
    <!doctype html>
    <title>Breast Cancer Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #fce4ec; color: #880e4f; text-align: center; padding: 20px; }
        h1 { color: #d81b60; margin-bottom: 0; }
        .banner { background-color: #d81b60; color: white; padding: 10px; margin-bottom: 20px; border-radius: 10px; display: flex; align-items: center; justify-content: center; }
        .ribbon { width: 50px; vertical-align: middle; margin: 0 10px; }
        form { background-color: #ffffff; padding: 20px; border-radius: 10px; display: inline-block; text-align: left; }
        .input-group { display: grid; grid-template-columns: repeat(10, 1fr); gap: 10px; }
        input[type="text"] { padding: 10px; border: 1px solid #d81b60; border-radius: 5px; width: 100%; }
        input[type="submit"] { background-color: #d81b60; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin-top: 10px; }
    </style>
    <div class="banner">
        <img src="https://i.etsystatic.com/9202327/r/il/e20a5a/2282030261/il_794xN.2282030261_g0tb.jpg" class="ribbon" alt="Pink Ribbon">
        <span>Ysais and Brian's Model</span>
        <img src="https://i.etsystatic.com/9202327/r/il/e20a5a/2282030261/il_794xN.2282030261_g0tb.jpg" class="ribbon" alt="Pink Ribbon">
    </div>
    <h1>Breast Cancer Prediction</h1>
    <form method="post">
        <div class="input-group">
            ''' + ''.join(
        [f'<input type="text" name="feature{i}" placeholder="Feature {i}" required>' for i in range(1, 31)]) + '''
        </div>
        <input type="submit" value="Predict">
    </form>
    '''


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
