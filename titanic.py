from subprocess import check_output
from speedml import Speedml

class TitanicML:
    def __init__(self):
        self.sml = None

    def run(self):
        print("running")
        self.data()
        self.models()
        #self.results()

    def data(self):
        print("preparing data")
        self.setup_speedml()
        self.prepare_data()

    def models(self):
        self.evaluate_models()
        #self.predict_models()

    def results(self):
        self.save_results()
        self.write_results()

    def prepare_data(self):
        self.strip_outliers()
        self.create_family_size()
        self.create_title()
        self.create_deck()
        self.map_sex()
        self.map_embarked()
        self.impute_values()
        self.create_ticket_density()
        self.create_age_density()
        self.create_fare_density()
        print("data prepared")

    def setup_speedml(self):
        print("Setting up Speedml")
        self.sml = Speedml('../input/train.csv', '../input/test.csv', target='Survived', uid='PassengerId')

    def strip_outliers(self):
        print("Stripping Outliers")
        self.sml.feature.outliers('Fare', upper=99)
        self.sml.feature.outliers('SibSp', upper=99)

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

    def map_sex(self):
        print("map sex")
        self.sml.feature.mapping('Sex', {'male': 0, 'female': 1})

    def create_deck(self):
        print("create deck and drop cabin")
        self.sml.feature.fillna(a='Cabin', new='Z')
        ## TODO: ^^ let's be smarter about this
        self.sml.feature.extract(new='Deck', a='Cabin', regex='([A-Z]){1}')
        self.sml.feature.labels(['Deck'])
        self.sml.feature.drop(['Cabin'])

    def map_embarked(self):
        print("map embarked")
        self.sml.feature.fillna(a='Embarked', new='Z')
        self.sml.feature.mapping('Embarked', {'S': 0, 'C': 1, 'Q': 2, 'Z': 3})

    def impute_values(self):
        print("IMPUTE BUT ONLY FOR NOW")
        self.sml.feature.impute()

    def create_ticket_density(self):
        print("drop ticket FOR NOW")
        self.sml.feature.density('Ticket')
        self.sml.feature.drop('Ticket')
        ## TODO: ^^ let's figure out Deck using this and PClass

    def create_fare_density(self):
        print("FOR NOW add Fare densities")
        self.sml.feature.density(['Fare'])

    def create_age_density(self):
        print("FOR NOW add Age densities")
        self.sml.feature.density(['Age'])

    def prepare_models(self):
        self.sml.model.data()
        self.refine_max_depth_and_min_child_weight()
        self.refine_learning_rate_and_subsample()

    def refine_max_depth_and_min_child_weight(self):
        select_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
        fixed_params = {'learning_rate': 0.1, 'subsample': 0.8,
                        'colsample_bytree': 0.8, 'seed':0,
                        'objective': 'binary:logistic'}
        self.sml.xgb.hyper(select_params, fixed_params)


    def refine_learning_rate_and_subsample(self):
        select_params = {'learning_rate': [0.3, 0.1, 0.01], 'subsample': [0.7, 0.8, 0.9]}
        fixed_params = {'max_depth': 5, 'min_child_weight': 7,
                        'colsample_bytree': 0.8, 'seed': 0,
                        'objective': 'binary:logistic'}
        self.sml.xgb.hyper(select_params, fixed_params)

    def assign_tuned_variables(self):
        tuned_params = {'learning_rate': 0.1, 'subsample': 0.8,
                        'max_depth': 3, 'min_child_weight': 1,
                        'seed':0, 'colsample_bytree': 0.8,
                        'objective': 'binary:logistic'}
        self.sml.xgb.cv(tuned_params)
        tuned_params['n_estimators'] = self.sml.xgb.cv_results.shape[0] - 1
        self.sml.xgb.params(tuned_params)

    def evaluate_models(self):
        """Show best models"""
        #self.sml.xgb.classifier()
        #self.sml.model.evaluate()
        #self.sml.plot.model_ranks()
        #self.sml.model.ranks()

    def predict_models(self):
        """predict and get accuracy"""
        self.sml.xgb.fit()
        self.sml.xgb.predict()
        self.sml.xgb.feature_selection()
        self.sml.xgb.sample_accuracy()

    def save_results(self):
        """save results when happy"""
        self.sml.save_results(
            columns={'PassengerId': self.sml.uid,
                     'Survived': self.sml.xgb.predictions},
            file_path='output/titanic-speedml-{}.csv'.format(self.sml.slug()))
        self.sml.slug()

    def write_results(self):
        print(check_output(["ls", "../input"]).decode("utf8"))