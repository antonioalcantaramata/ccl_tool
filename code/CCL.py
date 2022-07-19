import pandas as pd
import numpy as np
from sklearn.linear_model import QuantileRegressor
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVR
from utils import *
from dnn import *
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from pyomo import environ
from pyomo.environ import *


class CCL(object):
    '''
    Expected X, y and seek quantile for initialization.
    For methodology, one between ['point', 'quantile', 'superquantile'] 
    '''
    
    def __init__(self, X, y, methodology, p_model = None, q=None, M_super=None, side=None):
        if not methodology in ['point', 'quantile', 'superquantile']: #error!
            raise ValueError("Invalid methodology! Choose one between one between ['point', 'quantile', 'superquantile']") 
        
        if not p_model in ['lr', 'svm', 'tree', 'rf', 'gbm', 'nn']: #error!
            raise ValueError("Invalid prediction model! Choose one between one between ['lr', 'svm', 'tree', 'rf', 'gbm', 'nn']") 
        
        if (p_model == 'gbm') & (methodology == 'superquantile'): #error!
            raise ValueError("Superquantile estimation with GBM is not implemented yet.") 
        
        if (p_model == 'svm') & (methodology != 'point'): #error!
            raise ValueError("(Super)quantile estimation with linear SVM is not available in python. Pleaso, consider another predictive model.") 
        
        
        self.metho = methodology
        self.p_model = p_model
        self.X = X
        self.y = y
        if self.metho == 'quantile':
            self.q = q
        elif self.metho == 'superquantile':
            self.q = q
            self.M = M_super
            self.side = side
            if not side in ['left', 'right']: #error!
                raise ValueError("Invalid option for side of superquantile! Choose one between one between ['left', 'right']") 
    
    
    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=0)
        
        if self.metho == 'quantile':
            if self.p_model == 'lr':
                reg = sm.QuantReg(y_train, sm.add_constant(X_train, has_constant='add'))
                reg_fitted = reg.fit(q=self.q, max_iter=10000)
                preds = reg_fitted.predict(sm.add_constant(X_test, has_constant='add'))
                test_loss = quantile_loss(y_test, preds, self.q)
                
                print('Linear quantile regression fitted. Test quantile loss', test_loss)
                
                reg = sm.QuantReg(self.y, sm.add_constant(self.X, has_constant='add'))
                reg_fitted = reg.fit(q=self.q, max_iter=10000)
                
                return reg_fitted
            elif self.p_model == 'tree':
                tuning_results = []
                for max_d in [3,4,5,6,7,8,9,10]:
                    for min_leaf in [20, 40, 60, 80, 100]:
                        model = DecisionTreeRegressor(max_depth=max_d, min_samples_leaf=min_leaf, random_state=0)
                        model.fit(X_train, y_train)
                        
                        leaf_index = pd.DataFrame(model.apply(X_train), columns=['leaf_index'], index=y_train.index)
                        leaf_df = pd.concat([leaf_index, y_train], axis=1).groupby('leaf_index').agg(list).reset_index()
                        leaf_df.columns = ['leaf_index', 'leaf_values']
                        
                        leaf_df['leaf_size'] = leaf_df.leaf_values.apply(len)
                        quant = []
                        for l in leaf_df.leaf_values:
                            quant.append(np.quantile(l, self.q))
                        leaf_df['leaf_quant'] = quant
                        
                        preds = [float(leaf_df.loc[leaf_df['leaf_index'].isin(model.apply(X_test.iloc[[i]]))]['leaf_quant']) for i in range(X_test.shape[0])]
                        res_df = pd.DataFrame({'y': y_test, 'q_pred': preds})
                        
                        loss = quantile_loss(res_df.y, res_df.q_pred, q=self.q)
                        tuning_results.append([max_d, min_leaf, loss])
                res_df = pd.DataFrame(tuning_results, columns=['max_depth', 'min_samples_leaf', 'q_loss'])
                max_d = int(res_df[res_df.q_loss == res_df.q_loss.min()].iloc[[0]]['max_depth'])
                min_samp = int(res_df[res_df.q_loss == res_df.q_loss.min()].iloc[[0]]['min_samples_leaf'])
                test_loss = float(res_df[res_df.q_loss == res_df.q_loss.min()].iloc[[0]]['q_loss'])
                
                model = DecisionTreeRegressor(max_depth=max_d, min_samples_leaf=min_samp, random_state=0)
                model.fit(self.X, self.y)
                
                print('Regression tree for quantile estimation fitted.')
                print('Best hyper-parameter set: Max depth', max_d, 'Minimum samples in leaf', min_samp, 'with test quantile loss', test_loss)
                
                return model
            elif self.p_model == 'rf':
                tuning_results = []
                for max_d in [3,4,5,6,7,8,9,10,11,12]:
                    for min_leaf in [20, 30, 40, 50, 60]:
                        for n in [50, 100, 150]:
                            model = RandomForestQuantileRegressor(n_estimators=n, max_depth=max_d, min_samples_leaf=min_leaf, n_jobs=-1, random_state=0)
                            model.fit(X_train, y_train)
                            
                            rf_preds = model.predict(X_test, quantile=self.q * 100)
                            res_df = pd.DataFrame({'y': y_test, 'q_pred': rf_preds})
                            loss = quantile_loss(res_df.y, res_df.q_pred, q=self.q)
                            tuning_results.append([n, max_d, min_leaf, loss])
                res_df = pd.DataFrame(tuning_results, columns=['n_estimators','max_depth', 'min_samples_leaf', 'loss'])
                est = int(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['n_estimators'])
                max_d = int(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['max_depth'])
                min_samp = int(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['min_samples_leaf'])
                test_loss = float(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['loss'])
                model = RandomForestQuantileRegressor(n_estimators=est, max_depth=max_d, min_samples_leaf=min_samp, n_jobs=-1, random_state=0)
                model.fit(self.X, self.y)
                print('Random forest for quantile estimation fitted.')
                print('Best hyper-parameter set: Number of trees', est, 'Max depth', max_d, 'Minimum samples in leaf', min_samp, 'with test quantile loss', test_loss)
                return model
            elif self.p_model == 'gbm':
                tuning_results = []
                for max_d in [3,4,5,6,7]:
                    for min_leaf in [20, 40, 60, 80]:
                        for n in [20, 40, 60]:
                            for lr in [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]:
                                model = GradientBoostingRegressor(loss='quantile', alpha=self.q, n_estimators=n, max_depth=max_d, min_samples_leaf=min_leaf, learning_rate=lr, random_state=0)
                                model.fit(X_train, y_train)
                                preds = model.predict(X_test)
                                res_df = pd.DataFrame({'y': y_test, 'pred': preds})
                                loss = quantile_loss(res_df.y, res_df.pred, self.q)
                                tuning_results.append([lr, n, max_d, min_leaf, loss])
                res_df = pd.DataFrame(tuning_results, columns=['lr','n_estimators','max_depth', 'min_samples_leaf', 'loss'])
                lr = float(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['lr'])
                est = int(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['n_estimators'])
                max_d = int(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['max_depth'])
                min_samp = int(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['min_samples_leaf'])
                test_loss = float(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['loss'])
                model = GradientBoostingRegressor(loss='quantile', alpha=self.q, n_estimators=est, max_depth=max_d, min_samples_leaf=min_samp, learning_rate=lr, random_state=0)
                model.fit(self.X, self.y)
                print('Gradient boosting for quantile estimation fitted.')
                print('Best hyper-parameter set: Number of trees', est, 'Max depth', max_d, 'Minimum samples in leaf', min_samp, 'Learning rate', lr, 'with test quantile loss', test_loss)
                return model
            elif self.p_model == 'nn':
                nn_tool = dnn(self.X, self.y, self.metho, q=self.q, n_hidden=2, n_nodes=100, iters=6000)
                model = nn_tool.train()
                return model
        
        elif self.metho == 'superquantile':
            if self.p_model == 'lr':
                fitted_models = []
                if self.side == 'right':
                    delta = (1-self.q) / self.M
                    w = delta / (1-self.q)
                    quants = [ self.q + (j-0.5)*delta for j in range(1,self.M+1) ]
                else:
                    delta = (self.q) / self.M
                    w = delta / (self.q)
                    quants = [ self.q - (j-0.5)*delta for j in range(1,self.M+1) ]
                
                quant_losses = []
                for sq in quants:
                    reg = sm.QuantReg(self.y, sm.add_constant(self.X, has_constant='add'))
                    reg_fitted = reg.fit(q=sq, max_iter=10000)
                    preds = reg_fitted.predict(sm.add_constant(X_test, has_constant='add'))
                    test_loss = quantile_loss(y_test, preds, self.q)
                    quant_losses.append(test_loss)
                    fitted_models.append(reg_fitted)
                
                print('Superquantile linear model fitted from quantiles:', quants)
                print('Mean quantile loss:', np.mean(quant_losses))
                
                return fitted_models
            elif self.p_model == 'tree':
                tuning_results = []
                for max_d in [3,4,5,6,7,8,9,10]:
                    for min_leaf in [20, 40, 60, 80, 100]:
                        model = DecisionTreeRegressor(max_depth=max_d, min_samples_leaf=min_leaf, random_state=0)
                        model.fit(X_train, y_train)
                        
                        leaf_index = pd.DataFrame(model.apply(X_train), columns=['leaf_index'], index=y_train.index)
                        leaf_df = pd.concat([leaf_index, y_train], axis=1).groupby('leaf_index').agg(list).reset_index()
                        leaf_df.columns = ['leaf_index', 'leaf_values']
                        
                        leaf_df['leaf_size'] = leaf_df.leaf_values.apply(len)
                        quant = []
                        for l in leaf_df.leaf_values:
                            quant.append(np.quantile(l, self.q))
                        leaf_df['leaf_quant'] = quant
                        
                        preds = [float(leaf_df.loc[leaf_df['leaf_index'].isin(model.apply(X_test.iloc[[i]]))]['leaf_quant']) for i in range(X_test.shape[0])]
                        res_df = pd.DataFrame({'y': y_test, 'q_pred': preds})
                        
                        loss = quantile_loss(res_df.y, res_df.q_pred, q=self.q)
                        tuning_results.append([max_d, min_leaf, loss])
                res_df = pd.DataFrame(tuning_results, columns=['max_depth', 'min_samples_leaf', 'q_loss'])
                max_d = int(res_df[res_df.q_loss == res_df.q_loss.min()].iloc[[0]]['max_depth'])
                min_samp = int(res_df[res_df.q_loss == res_df.q_loss.min()].iloc[[0]]['min_samples_leaf'])
                test_loss = float(res_df[res_df.q_loss == res_df.q_loss.min()].iloc[[0]]['q_loss'])
                
                model = DecisionTreeRegressor(max_depth=max_d, min_samples_leaf=min_samp, random_state=0)
                model.fit(self.X, self.y)
                
                print('Regression tree for superquantile estimation fitted.')
                print('Best hyper-parameter set: Max depth', max_d, 'Minimum samples in leaf', min_samp, 'with test quantile loss', test_loss)
                
                return model
            elif self.p_model == 'rf':
                tuning_results = []
                for max_d in [3,4,5,6,7,8,9,10,11,12]:
                    for min_leaf in [20, 30, 40, 50, 60]:
                        for n in [50, 100, 150]:
                            model = RandomForestQuantileRegressor(n_estimators=n, max_depth=max_d, min_samples_leaf=min_leaf, n_jobs=-1, random_state=0)
                            model.fit(X_train, y_train)
                            
                            rf_preds = model.predict(X_test, quantile=self.q * 100)
                            res_df = pd.DataFrame({'y': y_test, 'q_pred': rf_preds})
                            loss = quantile_loss(res_df.y, res_df.q_pred, q=self.q)
                            tuning_results.append([n, max_d, min_leaf, loss])
                res_df = pd.DataFrame(tuning_results, columns=['n_estimators','max_depth', 'min_samples_leaf', 'loss'])
                est = int(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['n_estimators'])
                max_d = int(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['max_depth'])
                min_samp = int(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['min_samples_leaf'])
                test_loss = float(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['loss'])
                model = RandomForestQuantileRegressor(n_estimators=est, max_depth=max_d, min_samples_leaf=min_samp, n_jobs=-1, random_state=0)
                model.fit(self.X, self.y)
                print('Random forest for superquantile estimation fitted.')
                print('Best hyper-parameter set: Number of trees', est, 'Max depth', max_d, 'Minimum samples in leaf', min_samp, 'with test quantile loss', test_loss)
                return model
            elif self.p_model == 'gbm':
                tuning_results = []
                for max_d in [3,4,5,6,7]:
                    for min_leaf in [20, 40, 60, 80]:
                        for n in [20, 40, 60]:
                            for lr in [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]:
                                model = GradientBoostingRegressor(loss='quantile', alpha=self.q, n_estimators=n, max_depth=max_d, min_samples_leaf=min_leaf, learning_rate=lr, random_state=0)
                                model.fit(X_train, y_train)
                                preds = model.predict(X_test)
                                res_df = pd.DataFrame({'y': y_test, 'pred': preds})
                                loss = quantile_loss(res_df.y, res_df.pred, self.q)
                                tuning_results.append([lr, n, max_d, min_leaf, loss])
                res_df = pd.DataFrame(tuning_results, columns=['lr','n_estimators','max_depth', 'min_samples_leaf', 'loss'])
                lr = float(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['lr'])
                est = int(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['n_estimators'])
                max_d = int(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['max_depth'])
                min_samp = int(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['min_samples_leaf'])
                test_loss = float(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['loss'])
                model = GradientBoostingRegressor(loss='quantile', alpha=self.q, n_estimators=est, max_depth=max_d, min_samples_leaf=min_samp, learning_rate=lr, random_state=0)
                model.fit(self.X, self.y)
                print('Gradient Boosting for superquantile estimation fitted.')
                print('Best hyper-parameter set: Number of trees', est, 'Max depth', max_d, 'Minimum samples in leaf', min_samp, 'Learning rate', lr, 'with test quantile loss', test_loss)
                return model
            elif self.p_model == 'nn':
                nn_tool = dnn(self.X, self.y, self.metho, n_hidden=2, n_nodes=100, iters=6000, q=self.q, M_super=self.M, side=self.side)
                model = nn_tool.train()
                return model
    
        
        else:
            if self.p_model == 'lr':
                reg = LinearRegression()
                reg.fit(X_train, y_train)
                preds = reg.predict(X_test)
                test_loss = mean_absolute_error(y_test, preds)
                
                print('OLS Linear regression fitted. Test MAE', test_loss)
                
                reg.fit(self.X, self.y)
                
                return reg
            elif self.p_model == 'tree':
                tuning_results = []
                for max_d in [3,4,5,6,7,8,9,10]:
                    for min_leaf in [20, 40, 60, 80, 100]:
                        model = DecisionTreeRegressor(max_depth=max_d, min_samples_leaf=min_leaf, random_state=0)
                        model.fit(X_train, y_train)
                        
                        preds = model.predict(X_test)
                        res_df = pd.DataFrame({'y': y_test, 'pred': preds})
                        
                        loss = test_loss = mean_absolute_error(res_df.y, res_df.pred)
                        tuning_results.append([max_d, min_leaf, loss])
                res_df = pd.DataFrame(tuning_results, columns=['max_depth', 'min_samples_leaf', 'loss'])
                max_d = int(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['max_depth'])
                min_samp = int(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['min_samples_leaf'])
                test_loss = float(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['loss'])
                
                model = DecisionTreeRegressor(max_depth=max_d, min_samples_leaf=min_samp, random_state=0)
                model.fit(self.X, self.y)
                
                print('Regression tree for point estimation fitted.')
                print('Best hyper-parameter set: Max depth', max_d, 'Minimum samples in leaf', min_samp, 'with test MAE', test_loss)
                
                return model
            elif self.p_model == 'rf':
                tuning_results = []
                for max_d in [3,4,5,6,7,8,9,10,11,12]:
                    for min_leaf in [20, 30, 40, 50, 60]:
                        for n in [50, 100, 150]:
                            model = RandomForestRegressor(n_estimators=n, max_depth=max_d, min_samples_leaf=min_leaf, n_jobs=-1, random_state=0)
                            model.fit(X_train, y_train)
                            
                            preds = model.predict(X_test)
                            res_df = pd.DataFrame({'y': y_test, 'pred': preds})
                            loss = test_loss = mean_absolute_error(res_df.y, res_df.pred)
                            tuning_results.append([n, max_d, min_leaf, loss])
                res_df = pd.DataFrame(tuning_results, columns=['n_estimators','max_depth', 'min_samples_leaf', 'loss'])
                est = int(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['n_estimators'])
                max_d = int(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['max_depth'])
                min_samp = int(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['min_samples_leaf'])
                test_loss = float(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['loss'])
                model = RandomForestRegressor(n_estimators=est, max_depth=max_d, min_samples_leaf=min_samp, n_jobs=-1, random_state=0)
                model.fit(self.X, self.y)
                print('Random forest for point estimation fitted.')
                print('Best hyper-parameter set: Number of trees', est, 'Max depth', max_d, 'Minimum samples in leaf', min_samp, 'with test MAE', test_loss)
                return model
            elif self.p_model == 'gbm':
                tuning_results = []
                for max_d in [3,4,5,6,7]:
                    for min_leaf in [20, 40, 60, 80]:
                        for n in [20, 40, 60]:
                            for lr in [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]:
                                model = GradientBoostingRegressor(n_estimators=n, max_depth=max_d, min_samples_leaf=min_leaf, learning_rate=lr, random_state=0)
                                model.fit(X_train, y_train)
                                preds = model.predict(X_test)
                                res_df = pd.DataFrame({'y': y_test, 'pred': preds})
                                loss = mean_absolute_error(res_df.y, res_df.pred)
                                tuning_results.append([lr, n, max_d, min_leaf, loss])
                res_df = pd.DataFrame(tuning_results, columns=['lr','n_estimators','max_depth', 'min_samples_leaf', 'loss'])
                lr = float(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['lr'])
                est = int(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['n_estimators'])
                max_d = int(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['max_depth'])
                min_samp = int(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['min_samples_leaf'])
                test_loss = float(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['loss'])
                model = GradientBoostingRegressor(n_estimators=est, max_depth=max_d, min_samples_leaf=min_samp, learning_rate=lr, random_state=0)
                model.fit(self.X, self.y)
                print('Gradient Boosting for point estimation fitted.')
                print('Best hyper-parameter set: Number of trees', est, 'Max depth', max_d, 'Minimum samples in leaf', min_samp, 'Learning rate', lr, 'with test MAE', test_loss)
                return model
            elif self.p_model == 'nn':
                nn_tool = dnn(self.X, self.y, self.metho, n_hidden=2, n_nodes=100, iters=6000)
                model = nn_tool.train()
                return model
            elif self.p_model == 'svm':
                tuning_results = []
                for cost in [.1,1,5,10,25,50,75,100]:
                    model = LinearSVR(C=cost, max_iter = 1e6, dual=False, loss = 'squared_epsilon_insensitive', random_state=0)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    res_df = pd.DataFrame({'y': y_test, 'pred': preds})
                    loss = mean_absolute_error(res_df.y, res_df.pred)
                    tuning_results.append([cost, loss])
                
                res_df = pd.DataFrame(tuning_results, columns=['cost', 'loss'])
                cost = float(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['cost'])
                test_loss = float(res_df[res_df.loss == res_df.loss.min()].iloc[[0]]['loss'])
                model = LinearSVR(C=cost, max_iter = 1e5, dual=False, loss = 'squared_epsilon_insensitive', random_state=0)
                model.fit(self.X, self.y)
                print('LinearSVM for point estimation fitted.')
                print('Best hyper-parameter set: Cost', cost, 'with test MAE', test_loss)
                return model
    
    def constraint_build(self, fitted_model):
        if self.p_model == 'lr':
            if self.metho == 'quantile':
                columns = [feature for feature in self.X.columns]
                constraint = pd.DataFrame(data=[fitted_model.params[1:]], columns=columns)
                constraint['intercept'] = fitted_model.params[0]
                return constraint
            elif self.metho == 'superquantile':
                constraint = pd.DataFrame()
                columns = [feature for feature in self.X.columns]
                for i in range(self.M):
                    constraint_tmp = pd.DataFrame(data=[fitted_model[i].params[1:]], columns=columns)
                    constraint_tmp['intercept'] = fitted_model[i].params[0]
                    constraint = pd.concat([constraint,constraint_tmp], ignore_index=True)
                cons_sq = constraint.mean()
                constraint = pd.DataFrame(data=[cons_sq[:-1]], columns=columns)
                constraint['intercept'] = cons_sq[-1]
                return constraint
            else:
                columns = [feature for feature in self.X.columns]
                constraint = pd.DataFrame(data=[fitted_model.coef_], columns=columns)
                constraint['intercept'] = fitted_model.intercept_
                return constraint
        elif self.p_model == 'tree':
            children_left = fitted_model.tree_.children_left
            children_right = fitted_model.tree_.children_right
            feature = fitted_model.tree_.feature
            threshold = fitted_model.tree_.threshold
            
            leave_id = fitted_model.apply(self.X)
            columns = ['ID', 'leaf'] + [feature for feature in self.X.columns] + ['threshold', 'prediction']
            constraints = pd.DataFrame(columns=columns)
            
            for i, leaf in enumerate(np.unique(leave_id)):
                path_leaf = []
                find_path_skTree(0, path_leaf, leaf, children_left, children_right)
                constraints_leaf = get_rule_skTree(leaf, path_leaf, self.X.columns, columns, i+1, children_left, feature, threshold)
                constraints_leaf['prediction'] = fitted_model.tree_.value[leaf].item()
                constraints = pd.concat([constraints, constraints_leaf]) 
            
            leaf_index = pd.DataFrame(fitted_model.apply(self.X), columns=['leaf'], index=self.y.index)
            leaf_df = pd.concat([leaf_index, self.y], axis=1).groupby('leaf').agg(list).reset_index()
            leaf_df.columns = ['leaf', 'leaf_values']
            
            
            if self.metho == 'superquantile':
                quant = []
                for l in leaf_df.leaf_values:
                    quant.append(np.quantile(l, self.q))
                leaf_df['leaf_quant'] = quant
                sqs = []
                for i,l in enumerate(leaf_df.leaf_values):
                    if self.side == 'left':
                        tmp = filter(lambda vals: vals <= leaf_df.leaf_quant[i], l)
                    else: 
                        tmp = filter(lambda vals: vals >= leaf_df.leaf_quant[i], l)
                    tmp = list(tmp)
                    sq = np.mean(tmp)
                    sqs.append(sq)
                
                leaf_df['leaf_superq'] = sqs
                
                leaf_df = leaf_df.drop(['leaf_values','leaf_quant'], axis=1)
                constraints = pd.merge(constraints, leaf_df, 'left')
                constraints.drop(['prediction','leaf'], axis=1, inplace=True)
                constraints.rename(columns={'leaf_superq':'prediction'}, inplace=True)
                
                return constraints
            
            elif self.metho == 'quantile':
                quant = []
                for l in leaf_df.leaf_values:
                    quant.append(np.quantile(l, self.q))
                leaf_df['leaf_quant'] = quant
                leaf_df = leaf_df.drop(['leaf_values'], axis=1)
                constraints = pd.merge(constraints, leaf_df, 'left')
                constraints.drop(['prediction','leaf'], axis=1, inplace=True)
                constraints.rename(columns={'leaf_quant':'prediction'}, inplace=True)
                
                return constraints
            else:
                constraints.drop(['leaf'], axis=1, inplace=True)
                return constraints
        elif self.p_model == 'rf':
            columns = ['Tree_id', 'ID'] + [feature for feature in self.X.columns] + ['threshold', 'prediction']
            leaves = pd.DataFrame()
            constraints = pd.DataFrame(columns=columns)
            for tree_id, tree in enumerate(fitted_model):
                children_left = tree.tree_.children_left
                children_right = tree.tree_.children_right
                feature = tree.tree_.feature
                threshold = tree.tree_.threshold
                leave_id = tree.apply(self.X)
                for i, leaf in enumerate(np.unique(leave_id)):
                    path_leaf = []
                    find_path_skTree(0, path_leaf, leaf, children_left, children_right)
                    constraints_leaf = get_rule_skTree(leaf, path_leaf, self.X.columns, columns, i+1, children_left, feature, threshold)
                    constraints_leaf['Tree_id'] = tree_id
                    constraints_leaf['prediction'] = tree.tree_.value[leaf].item()
                    constraints = pd.concat([constraints, constraints_leaf])
                
                leaf_index = pd.DataFrame(tree.apply(self.X), columns=['leaf'], index=self.y.index)
                leaf_df = pd.concat([leaf_index, self.y], axis=1).groupby('leaf').agg(list).reset_index()
                leaf_df.columns = ['leaf', 'leaf_values']
                leaf_df['Tree_id'] = tree_id
                
                if self.metho == 'quantile':
                    quant = []
                    for l in leaf_df.leaf_values:
                        quant.append(np.quantile(l, self.q))
                    leaf_df['leaf_quant'] = quant
                    leaves = pd.concat([leaves, leaf_df])
                    
                elif self.metho == 'superquantile':
                    quant = []
                    for l in leaf_df.leaf_values:
                        quant.append(np.quantile(l, self.q))
                    leaf_df['leaf_quant'] = quant
                    sqs = []
                    for i,l in enumerate(leaf_df.leaf_values):
                        if self.side == 'left':
                            tmp = filter(lambda vals: vals <= leaf_df.leaf_quant[i], l)
                        else: 
                            tmp = filter(lambda vals: vals >= leaf_df.leaf_quant[i], l)
                        tmp = list(tmp)
                        sq = np.mean(tmp)
                        sqs.append(sq)
                    leaf_df['leaf_superq'] = sqs
                    leaves = pd.concat([leaves, leaf_df])
            
            if self.metho == 'quantile':
                constraints = pd.merge(constraints, leaves, 'left', on=['Tree_id','leaf'])
                constraints.drop(['prediction','leaf','leaf_values'], axis=1, inplace=True)
                constraints.rename(columns={'leaf_quant':'prediction'}, inplace=True)
            elif self.metho == 'superquantile':
                constraints = pd.merge(constraints, leaves, 'left', on=['Tree_id','leaf'])
                constraints.drop(['prediction','leaf','leaf_values','leaf_quant'], axis=1, inplace=True)
                constraints.rename(columns={'leaf_superq':'prediction'}, inplace=True)
            
            return constraints
        elif self.p_model == 'gbm':
            leaves = pd.DataFrame()
            columns = ['Tree_id', 'ID'] + [feature for feature in self.X.columns] + ['threshold', 'prediction', 'initial_prediction', 'learning_rate']
            constraints = pd.DataFrame(columns=columns)
            for tree_id, tree_array in enumerate(fitted_model.estimators_):
                tree = tree_array.item()
                children_left = tree.tree_.children_left
                children_right = tree.tree_.children_right
                feature = tree.tree_.feature
                threshold = tree.tree_.threshold
                leave_id = tree.apply(self.X)
                
                if self.metho != 'superquantile':
                    for i, leaf in enumerate(np.unique(leave_id)):
                        path_leaf = []
                        find_path_skTree(0, path_leaf, leaf, children_left, children_right)
                        constraints_leaf = get_rule_skTree(leaf, path_leaf, self.X.columns, columns[:-1], i + 1, children_left, feature, threshold)
                        constraints_leaf['Tree_id'] = tree_id
                        constraints_leaf['prediction'] = tree.tree_.value[leaf].item()
                        constraints_leaf['initial_prediction'] = fitted_model.init_.constant_.item()
                        constraints_leaf['learning_rate'] = fitted_model.learning_rate
                        constraints = pd.concat([constraints, constraints_leaf])
                else:
                    for i, leaf in enumerate(np.unique(leave_id)):
                        path_leaf = []
                        find_path_skTree(0, path_leaf, leaf, children_left, children_right)
                        constraints_leaf = get_rule_skTree(leaf, path_leaf, self.X.columns, columns[:-1], i + 1, children_left, feature, threshold)
                        constraints_leaf['Tree_id'] = tree_id
                        constraints_leaf['prediction'] = tree.tree_.value[leaf].item()
                        constraints_leaf['initial_prediction'] = fitted_model.init_.constant_.item()
                        constraints_leaf['learning_rate'] = fitted_model.learning_rate
                        if tree_id == (fitted_model.n_estimators_ - 1):
                            leaf_index = pd.DataFrame(tree.apply(self.X), columns=['leaf'], index=self.y.index)
                            leaf_df = pd.concat([leaf_index, self.y], axis=1).groupby('leaf').agg(list).reset_index()
                            leaf_df.columns = ['leaf', 'leaf_values']
                            leaf_df['Tree_id'] = tree_id
                            quant = []
                            for l in leaf_df.leaf_values:
                                quant.append(np.quantile(l, self.q))
                            leaf_df['leaf_quant'] = quant
                            sqs = []
                            for i,l in enumerate(leaf_df.leaf_values):
                                if self.side == 'left':
                                    tmp = filter(lambda vals: vals <= leaf_df.leaf_quant[i], l)
                                else: 
                                    tmp = filter(lambda vals: vals >= leaf_df.leaf_quant[i], l)
                                tmp = list(tmp)
                                sq = np.mean(tmp)
                                sqs.append(sq)
                            leaf_df['leaf_superq'] = sqs
                            #leaves = pd.concat([leaves, leaf_df])
                            
                            #leaf_df = leaf_df.drop(['leaf_values','leaf_quant'], axis=1)
                            constraints_leaf = pd.merge(constraints_leaf, leaf_df, 'left')
                            constraints_leaf.drop(['prediction','leaf_values','leaf_quant'], axis=1, inplace=True)
                            constraints_leaf.rename(columns={'leaf_superq':'prediction'}, inplace=True)
                            #constraints_leaf['prediction'] = sqs
                            constraints = pd.concat([constraints, constraints_leaf])
                        else:
                            constraints = pd.concat([constraints, constraints_leaf])
                        
            
            #if self.metho == 'superquantile':
            #    constraints.drop(['prediction','leaf','leaf_values','leaf_quant'], axis=1, inplace=True)
            #    constraints.rename(columns={'leaf_superq':'prediction'}, inplace=True)
            constraints.drop(['leaf'], axis=1, inplace=True)
            return constraints
        elif self.p_model == 'nn':
            weight_names = []
            weight_values = []
            bias_names = []
            bias_values = []
            for name, param in fitted_model.named_parameters():
                if 'weight' in name:
                    weight_names.append(name)
                    weight_values.append(param.detach().numpy())
                else:
                    bias_names.append(name)
                    bias_values.append(param.detach().numpy())
            constraints = constraint_extrapolation_MLP(weight_values,bias_values,weight_names)
            
            return constraints
        elif self.p_model == 'svm':
            columns = [feature for feature in self.X.columns]
            constraint = pd.DataFrame(data=[fitted_model.coef_], columns=columns)
            constraint['intercept'] = fitted_model.intercept_
            
            return constraint
    
    def const_embed(self, opt_model, constaints, outcome, lb=None, ub=None):
        '''
        This function embdeds a fitted prediction model within the optimization problem.
        Expecting a defined optimization model "opt_model", the ccl tool and constraint dataframe "constraints".
        A lower bound (lb) or upper bound (ub) is also expected.
        '''
        
        # Predefining variable y
        opt_model.y = Var(Any, dense=False, domain=Reals)
        
        ### For linear models
        if (self.p_model == 'lr') | (self.p_model == 'svm'):
            intercept = constaints['intercept'][0]
            coeff = constaints.drop(['intercept'], axis=1, inplace=False).loc[0, :]
            opt_model.add_component('LR'+outcome, Constraint(expr=opt_model.y[outcome] == sum(opt_model.x[i] * coeff.loc[i] for i in pd.DataFrame(coeff).index) + intercept))
            if not pd.isna(ub):
                opt_model.add_component('ub_' + outcome, Constraint(expr=opt_model.y[outcome] <= ub))
            if not pd.isna(lb):
                opt_model.add_component('lb_' + outcome, Constraint(expr=opt_model.y[outcome] >= lb))
        
        ### For trees
        if self.p_model == 'tree':
            M = 1e5
            opt_model.l = Var(Any, dense=False, domain=Binary)
            leaf_values = constaints.loc[:, ['ID', 'prediction']].drop_duplicates().set_index('ID')
            intercept = constaints['threshold']
            coeff = constaints.drop(['ID', 'threshold', 'prediction'], axis=1, inplace=False).reset_index(drop=True)
            l_ids = constaints['ID']
            n_constr = coeff.shape[0]
            L = np.unique(constaints['ID'])
            def constraintsTree_1(model, j):
                return sum(model.x[i]*coeff.loc[j, i] for i in self.X.columns) <= intercept.iloc[j] + M*(1-model.l[(outcome,str(l_ids.iloc[j]))])
            def constraintsTree_2(model):
                return sum(model.l[(outcome, str(i))] for i in L) == 1
            def constraintTree(model):
                return model.y[outcome] == sum(leaf_values.loc[i, 'prediction'] * model.l[(outcome, str(i))] for i in L)
            
            opt_model.add_component(outcome+'_1', Constraint(range(n_constr), rule=constraintsTree_1))
            opt_model.add_component('DT'+outcome, Constraint(rule=constraintTree))
            opt_model.add_component(outcome+'_2', Constraint(rule=constraintsTree_2))
            
            if not pd.isna(ub):
                opt_model.add_component('ub_' + outcome, Constraint(expr=opt_model.y[outcome] <= ub))
            if not pd.isna(lb):
                opt_model.add_component('lb_' + outcome, Constraint(expr=opt_model.y[outcome] >= lb))
        
        ### For RF
        if self.p_model == 'rf':
            M = 1e5
            opt_model.l = Var(Any, dense=False, domain=Binary)
            constaints['Tree_id'] = [outcome + '_' + str(i) for i in constaints['Tree_id']]
            T = np.unique(constaints['Tree_id'])
            for i, t in enumerate(T):
                tree_table = constaints.loc[constaints['Tree_id'] == t, :].drop('Tree_id', axis=1)
                # don't set LB, UB, or objective for individual trees
                #constraints_tree(model, t, tree_table, lb=None, ub=None, weight_objective=0, SCM=None, features=features)
                leaf_values = tree_table.loc[:, ['ID', 'prediction']].drop_duplicates().set_index('ID')
                intercept = tree_table['threshold']
                coeff = tree_table.drop(['ID', 'threshold', 'prediction'], axis=1, inplace=False).reset_index(drop=True)
                l_ids = tree_table['ID']
                n_constr = coeff.shape[0]
                L = np.unique(tree_table['ID'])
                def constraintsTree_1(model, j):
                    return sum(model.x[i]*coeff.loc[j, i] for i in self.X.columns) <= intercept.iloc[j] + M*(1-model.l[(t,str(l_ids.iloc[j]))])
                def constraintsTree_2(model):
                    return sum(model.l[(t, str(i))] for i in L) == 1
                def constraintTree(model):
                    return model.y[t] == sum(leaf_values.loc[i, 'prediction'] * model.l[(t, str(i))] for i in L)
                
                opt_model.add_component(t+'_1', Constraint(range(n_constr), rule=constraintsTree_1))
                opt_model.add_component('DT'+t, Constraint(rule=constraintTree))
                opt_model.add_component(t+'_2', Constraint(rule=constraintsTree_2))
            
            opt_model.add_component('RF'+outcome, Constraint(rule=opt_model.y[outcome] == 1 / len(T) * quicksum(opt_model.y[j] for j in T)))
            if not pd.isna(ub):
                opt_model.add_component('ub_' + outcome, Constraint(expr=opt_model.y[outcome] <= ub))
            if not pd.isna(lb):
                opt_model.add_component('lb_' + outcome, Constraint(expr=opt_model.y[outcome] >= lb))
            
        ### for gbm
        if self.p_model == 'gbm':
            M = 1e5
            opt_model.l = Var(Any, dense=False, domain=Binary)
            constaints['Tree_id'] = [outcome + '_' + str(i) for i in constaints['Tree_id']]
            T = np.unique(constaints['Tree_id'])
            for i, t in enumerate(T):
                tree_table = constaints.loc[constaints['Tree_id'] == t, :].drop(['Tree_id', 'initial_prediction', 'learning_rate'], axis=1, inplace=False)
                # don't set LB, UB, or objective for individual trees
                #constraints_tree(model, t, tree_table, lb=None, ub=None, weight_objective=0, SCM=None, features=features)
                leaf_values = tree_table.loc[:, ['ID', 'prediction']].drop_duplicates().set_index('ID')
                intercept = tree_table['threshold']
                coeff = tree_table.drop(['ID', 'threshold', 'prediction'], axis=1, inplace=False).reset_index(drop=True)
                l_ids = tree_table['ID']
                n_constr = coeff.shape[0]
                L = np.unique(tree_table['ID'])
                def constraintsTree_1(model, j):
                    return sum(model.x[i]*coeff.loc[j, i] for i in self.X.columns) <= intercept.iloc[j] + M*(1-model.l[(t,str(l_ids.iloc[j]))])
                def constraintsTree_2(model):
                    return sum(model.l[(t, str(i))] for i in L) == 1
                def constraintTree(model):
                    return model.y[t] == sum(leaf_values.loc[i, 'prediction'] * model.l[(t, str(i))] for i in L)
                
                opt_model.add_component(t+'_1', Constraint(range(n_constr), rule=constraintsTree_1))
                opt_model.add_component('DT'+t, Constraint(rule=constraintTree))
                opt_model.add_component(t+'_2', Constraint(rule=constraintsTree_2))
            def constraint_gbm(opt_model):
                return opt_model.y[outcome] == np.unique(constaints['initial_prediction']).item() + np.unique(constaints['learning_rate']).item() * quicksum(opt_model.y[j] for j in T)
            opt_model.add_component('GBM'+outcome, Constraint(rule=constraint_gbm))
            if not pd.isna(ub):
                opt_model.add_component('ub_' + outcome, Constraint(expr=opt_model.y[outcome] <= ub))
            if not pd.isna(lb):
                opt_model.add_component('lb_' + outcome, Constraint(expr=opt_model.y[outcome] >= lb))
        
        ### for nn
        elif self.p_model == 'nn':
            M_l=-1e5
            M_u=1e5
            opt_model.v = Var(Any, dense=False, domain=NonNegativeReals)
            opt_model.v_ind = Var(Any, dense=False, domain=Binary)
            nodes_input = range(len(self.X.columns))
            v_input = [opt_model.x[i] for i in self.X.columns]
            max_layer = max(constaints['layer'])
            
            for l in range(max_layer + 1):
                df_layer = constaints.query('layer == %d' % l)
                max_nodes = [k for k in df_layer.columns if 'node_' in k]
                # coeffs_layer = np.array(df_layer.iloc[:, range(len(max_nodes))].dropna(axis=1))
                coeffs_layer = np.array(df_layer.loc[:, max_nodes].dropna(axis=1))
                intercepts_layer = np.array(df_layer['intercept'])
                nodes = df_layer['node']
                
                if l == max_layer:
                    if self.metho != 'superquantile':
                        node = nodes.iloc[0]  # only one node in last layer
                        opt_model.add_component('MLP'+outcome, Constraint(rule=opt_model.y[outcome] == sum(v_input[i] * coeffs_layer[node, i] for i in nodes_input) + intercepts_layer[node]))
                    else:
                        for node in nodes:
                            opt_model.add_component('MLP'+'q'+str(node), Constraint(rule=opt_model.y[node] == sum(v_input[i] * coeffs_layer[node, i] for i in nodes_input) + intercepts_layer[node]))
                        opt_model.add_component('MLP'+outcome, Constraint(rule=opt_model.y[outcome] == 1 / len(nodes) * quicksum(opt_model.y[node] for node in nodes)))
                else:
                    # Save v_pos for input to next layer
                    v_pos_list = []
                    for node in nodes:
                        ## Initialize variables
                        v_pos_list.append(opt_model.v[(outcome, l, node)])
                        opt_model.add_component('constraint_1_' + str(l) + '_'+str(node)+outcome,
                                            Constraint(rule=opt_model.v[(outcome, l, node)] >= sum(v_input[i] * coeffs_layer[node, i] for i in nodes_input) + intercepts_layer[node]))
                        opt_model.add_component('constraint_2_' + str(l)+'_' + str(node) + outcome,
                                            Constraint(rule=opt_model.v[(outcome, l, node)] <= M_u * (opt_model.v_ind[(outcome, l, node)])))
                        opt_model.add_component('constraint_3_' + str(l)+'_' + str(node) + outcome,
                                            Constraint(rule=opt_model.v[(outcome, l, node)] <= sum(v_input[i] * coeffs_layer[node, i] for i in nodes_input) + intercepts_layer[node] - M_l * (1 - opt_model.v_ind[(outcome, l, node)])))
                    ## Prepare nodes_input for next layer
                    nodes_input = nodes
                    v_input = v_pos_list
            
            if not pd.isna(ub):
                opt_model.add_component('ub_' + outcome, Constraint(expr=opt_model.y[outcome] <= ub))
            if not pd.isna(lb):
                opt_model.add_component('lb_' + outcome, Constraint(expr=opt_model.y[outcome] >= lb))
        

