from flask import Flask
from flask_restx import Api, Resource, reqparse, inputs
# from flask_restful import  reqparse
import joblib
import numpy as np

APP = Flask(__name__)
API = Api(APP)

DIABETES_MODEL = joblib.load('diabetes.mdl')
name_space = API.namespace('main', description='Diabetes APIs')

# class Predict(Resource):
#
#     @staticmethod
#     def post():

parser = reqparse.RequestParser()
parser.add_argument('Pregnancies',type=inputs.int_range(0,40),
                    help="Number of Pregnancies",
                    default=2, required=True)
parser.add_argument('Glucose',type=inputs.int_range(1,300),
                    help="Level of glucose.",
                    default=150, required=True)
parser.add_argument('SkinThickness',type=inputs.int_range(1,100),
                    help="Skin Thickness.",
                    default=20, required=True)
parser.add_argument('BMI',type=float,
                    help="BMI level",
                    default=20.0, required=True)
parser.add_argument('Age',type=inputs.int_range(0,200),
                    help="How old is the patient.",
                    default=25, required=True)
@API.route('/predict')
class Predict(Resource):
    @API.doc('Checks if a monthly report exists')
    @API.expect(parser)
    def post(self):
        args = parser.parse_args()  # creates dict

        X_new = np.fromiter(args.values(), dtype=float)  # convert input to array

        out = {'Prediction': DIABETES_MODEL.predict([X_new])[0]}
        return out, 200



if __name__ == '__main__':
    APP.run(debug=True, port='1080')