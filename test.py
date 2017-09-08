from speedml import Speedml
sml = Speedml('../test/train.csv','../test/test.csv',target='Survived',uid='PassengerId')
print("Speedml set up!")

# def data_checkup():
#     sml.eda()
#     sml.plot.correlate()
#     sml.plot.distribute()
#     return
# data_checkup()

sml.feature.outliers('Fare', upper=99)
sml.feature.outliers('SibSp', upper=99)

# merge Parch and SibSp into FamilySize
sml.feature.sum(new='FamilySize', a='Parch', b='SibSp')
sml.feature.add('FamilySize', 1)
sml.feature.drop('Parch')
sml.feature.drop('SibSp')

# extract Title from Name
sml.feature.extract(new='Title', a='Name', regex=r" ([A-Za-z]+)\.")
sml.feature.replace(a='Title', match=['Lady', 'Countess', 'Dona', 'Mme'], new='Mrs')
sml.feature.replace(a='Title', match=['Don', 'Sir', 'Jonkheer'], new='Mr')
sml.feature.replace(a='Title', match=['Capt', 'Col', 'Dr', 'Major', 'Rev'], new='Crew')
sml.feature.replace(a='Title', match=['Mlle','Ms'], new='Miss')
sml.feature.mapping('Title', {'Miss': 1, 'Master': 2, 'Mrs': 3, 'Mr': 4, 'Crew': 5})
sml.feature.fillna(a='Title', new=0)
sml.feature.drop('Name')

# map sex
sml.feature.mapping('Sex', {'male': 0, 'female': 1})

# remap to deck
sml.feature.fillna(a='Cabin', new='Z')
## TODO: ^^ let's be smarter about this
sml.feature.extract(new='Deck', a='Cabin', regex='([A-Z]){1}')
sml.feature.labels('Deck')
sml.feature.drop(['Cabin'])

# map embarked
sml.feature.fillna(a='Embarked', new='Z')
sml.feature.mapping('Embarked', {'S': 0, 'C': 1, 'Q': 2, 'Z': 3})

# IMPUTE BUT ONLY FOR NOW
sml.feature.impute()

# drop ticket for now
sml.feature.density('Ticket')
sml.feature.drop('Ticket')
## TODO: ^^ let's figure out Deck using this and PClass

## FOR NOW add Age and Fare densities
sml.feature.density(['Age','Fare'])


## WE'RE READY
sml.eda()


### XGB STUFF - I don't know what this does yet
sml.model.data()
select_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
fixed_params = {'learning_rate': 0.1, 'subsample': 0.8,
                'colsample_bytree': 0.8, 'seed':0,
                'objective': 'binary:logistic'}
sml.xgb.hyper(select_params, fixed_params)


select_params = {'learning_rate': [0.3, 0.1, 0.01], 'subsample': [0.7,0.8,0.9]}
fixed_params = {'max_depth': 3, 'min_child_weight': 1,
                'colsample_bytree': 0.8, 'seed':0,
                'objective': 'binary:logistic'}
sml.xgb.hyper(select_params, fixed_params)

tuned_params = {'learning_rate': 0.1, 'subsample': 0.8,
                'max_depth': 3, 'min_child_weight': 1,
                'seed':0, 'colsample_bytree': 0.8,
                'objective': 'binary:logistic'}
sml.xgb.cv(tuned_params)

tuned_params['n_estimators'] = sml.xgb.cv_results.shape[0] - 1
sml.xgb.params(tuned_params)


## Show best models
sml.xgb.classifier()
sml.model.evaluate()
sml.plot.model_ranks()
sml.model.ranks()

## predict and get accuracy
sml.xgb.fit()
sml.xgb.predict()

sml.xgb.feature_selection()
sml.xgb.sample_accuracy()

