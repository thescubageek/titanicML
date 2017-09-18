## I freakin suck at Python right now but hopefully this will get better quickly

from subprocess import check_output
from speedml import Speedml
import warnings
warnings.filterwarnings('ignore')

class TitanicML:
    def __init__(self):
        self.sml = None

    def run(self):
        print("running")
        self.data()
        self.models()
        self.results()

    def data(self):
        self.setup_speedml()
        self.prepare_data()

    def models(self):
        self.prepare_models()
        self.evaluate_models()
        self.predict_models()

    def results(self):
        self.save_results()
        self.write_results()

    def prepare_data(self):
        print("preparing data")
        self.strip_outliers()
        self.create_features()
        self.map_features()
        self.impute_features()
        self.feature_densities()
        self.drop_features()
        print("data prepared")

    def setup_speedml(self):
        print("Setting up Speedml")
        self.sml = Speedml('../input/train.csv', '../input/test.csv', target='Survived', uid='PassengerId')

    ################ DATA PREPARATION  ################

    def strip_outliers(self):
        print("Stripping Outliers")
        self.sml.feature.outliers('Fare', upper=99)
        self.sml.feature.outliers('SibSp', upper=98)

    def create_features(self):
        self.create_family_size()
        self.create_title()
        self.create_deck()

    def map_features(self):
        self.map_sex()
        self.map_embarked()

    def impute_features(self):
        self.impute_ages()
        self.impute_values()

    def feature_densities(self):
        self.create_ticket_density()
        self.create_age_density()
        self.create_fare_density()

    def drop_features(self):
        self.drop_cabin()
        self.drop_ticket()
        #self.drop_fare()
        #self.drop_age()
        self.drop_embarked()

    ### CREATE FEATUES

    def create_family_size(self):
        print("Merge Parch and SibSp into FamilySize")
        self.sml.feature.sum(new='FamilySize', a='Parch', b='SibSp')
        self.sml.feature.add('FamilySize', 1)
        self.sml.feature.drop('Parch')
        self.sml.feature.drop('SibSp')

    def create_title(self):
        print("extract Title from Name")
        self.sml.feature.extract(new='Title', a='Name', regex=r" ([A-Za-z]+)\.")
        self.sml.feature.replace(a='Title', match=['Lady', 'Countess', 'Dona', 'Mme'], new='Mrs')
        self.sml.feature.replace(a='Title', match=['Don', 'Sir', 'Jonkheer'], new='Mr')
        self.sml.feature.replace(a='Title', match=['Capt', 'Col', 'Dr', 'Major', 'Rev'], new='Crew')
        self.sml.feature.replace(a='Title', match=['Mlle','Ms'], new='Miss')
        self.sml.feature.mapping('Title', {'Miss': 1, 'Master': 2, 'Mrs': 3, 'Mr': 4, 'Crew': 5})
        self.sml.feature.fillna(a='Title', new=0)
        self.sml.feature.drop('Name')

    def create_deck(self):
        print("create deck")
        self.sml.feature.fillna(a='Cabin', new='Z')
        self.sml.feature.extract(new='Deck', a='Cabin', regex='([A-Z]){1}')
        self.sml.feature.labels(['Deck', 'Cabin'])
        ## TODO ^^ figure out Deck from TicketPrice, Cabin, and PClass

    ### MAP FEATURES

    def map_sex(self):
        print("map sex")
        self.sml.feature.mapping('Sex', {'male': 0, 'female': 1})

    def map_embarked(self):
        print("map embarked")
        self.sml.feature.fillna(a='Embarked', new='Z')
        self.sml.feature.mapping('Embarked', {'S': 0, 'C': 1, 'Q': 2, 'Z': 3})

    ### IMPUTE FEATURES

    def impute_ages(self):
        print("Impute ages")
        titanic.sml.feature.fillna('Age',0)
        for df in [titanic.sml.train, titanic.sml.test]:
          for i in list(range(1,6)):
              titles = df[(df['Title'] == i) & (df['Age'] != 0)]
              title_mean_age = titles['Age'].mean()
              null_ages = df[(df['Title'] == i) & (df['Age'] == 0)]
              null_ages['Age'] = title_mean_age

    def impute_values(self):
        print("Impute remaining empty fields")
        self.sml.feature.impute()

    ### FEATURE DENSITIES

    def create_ticket_density(self):
        print("create ticket density")
        self.sml.feature.density('Ticket')
        ## TODO: ^^ let's figure out Deck using this and PClass

    def create_age_density(self):
        print("FOR NOW add Age densities")
        self.sml.feature.density(['Age'])

    def create_fare_density(self):
        print("FOR NOW add Fare densities")
        self.sml.feature.density(['Fare'])

    def drop_ticket(self):
        print("drop ticket")
        self.sml.feature.drop('Ticket')

    ### DROP FEATURES

    def drop_cabin(self):
        print("drop cabin")
        self.sml.feature.drop('Cabin')

    def drop_fare(self):
        print("drop fare")
        self.sml.feature.drop('Fare')

    def drop_age(self):
        print("drop age")
        self.sml.feature.drop('Age')

    def drop_embarked(self):
        print("drop embarked")
        self.sml.feature.drop('Embarked')

     ################ MODEL PREPARATION  ################

    def prepare_models(self):
        print("prepare models")
        self.sml.model.data()
        self.set_model_parameters()

    def set_model_parameters(self):
        print("set model parameters")
        ret1 = self.refine_max_depth_and_min_child_weight()
        ret2 = self.refine_learning_rate_and_subsample(ret1)
        self.assign_tuned_variables(ret1, ret2)

    def refine_max_depth_and_min_child_weight(self):
        """Finds best max_depth and min_child_weight against fixed params"""
        print("refine max depth and min child weight")
        select_params = {'max_depth': list(range(3, 9)), 'min_child_weight': list(range(1, 7))}
        fixed_params = {'learning_rate': 0.1, 'subsample': 0.8,
                        'colsample_bytree': 0.8, 'seed': 0,
                        'objective': 'binary:logistic'}
        ret = self.sml.xgb.hyper(select_params, fixed_params)
        return ret['params'][0]


    def refine_learning_rate_and_subsample(self, results):
        """Finds best refine_learning_rate and subsample against fixed params"""
        print("refine learning rate and subsamples with max_depth")
        learning_rate_range = [0.3,0.2,0.1,0.05,0.01]
        subsample_range = list(map(lambda x: str(x/10), range(6, 9)))
        select_params = {'learning_rate': learning_rate_range, 'subsample': subsample_range}
        fixed_params = {'max_depth': results['max_depth'], 'min_child_weight': results['min_child_weight'],
                        'colsample_bytree': 0.8, 'seed': 0,
                        'objective': 'binary:logistic'}
        ret = self.sml.xgb.hyper(select_params, fixed_params)
        return ret['params'][0]

    def assign_tuned_variables(self, ret1, ret2):
        """Assigned best fit params to XGBoost"""
        print("assign tuned variables")
        print("learning_rate: "+str(ret2['learning_rate']))
        print("subsample: "+str(ret2['subsample']))
        print("max_depth: "+str(ret1['max_depth']))
        print("min_child_weight: "+str(ret1['min_child_weight']))
        tuned_params = {'learning_rate': ret2['learning_rate'], 'subsample': ret2['subsample'],
                        'max_depth': ret1['max_depth'], 'min_child_weight': ret1['min_child_weight'],
                        'seed':0, 'colsample_bytree': 0.8,
                        'objective': 'binary:logistic'}
        self.sml.xgb.cv(tuned_params)
        tuned_params['n_estimators'] = self.sml.xgb.cv_results.shape[0] - 1
        self.sml.xgb.params(tuned_params)

    def evaluate_models(self):
        print("Show best models")
        self.sml.xgb.classifier()
        self.sml.model.evaluate()
        self.sml.plot.model_ranks()
        self.sml.model.ranks()

    def predict_models(self):
        print("predict and get accuracy")
        self.sml.xgb.fit()
        self.sml.xgb.predict()
        self.sml.xgb.feature_selection()
        self.sml.xgb.sample_accuracy()

     ################ RESULTS  ################

    def save_results(self):
        print("save results when happy")
        self.sml.save_results(
            columns={'PassengerId': self.sml.uid,
                     'Survived': self.sml.xgb.predictions},
            file_path='output/titanic-speedml-{}.csv'.format(self.sml.slug()))
        self.sml.slug()

    def write_results(self):
        print(check_output(["ls", "../input"]).decode("utf8"))