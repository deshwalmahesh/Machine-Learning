from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC,NuSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier,\
    GradientBoostingClassifier,RandomForestClassifier


ridge = RidgeClassifier()
logi = LogisticRegression()
svc = SVC()
nusvc=NuSVC()
gauss = GaussianNB()
bernoli = BernoulliNB()
ada = AdaBoostClassifier()
bag = BaggingClassifier()
extra = ExtraTreesClassifier()
grad = GradientBoostingClassifier()
forest = RandomForestClassifier()


classifiers = [ridge,logi,svc,nusvc,gauss,bernoli,ada,bag,extra,grad,forest]
classi_mod_with_coef_or_feat_imp= [forest,ridge,logi,bernoli,ada,extra,grad]
classifiers_names = [(type(model).__name__) for model in classifiers]


# Feature Selection Helper Functions and plotting


def plot_results(result_dict,x_features,rows=4,cols=3,total=11,):
    keys = list(result.keys())
    rows,cols,total= rows,cols,total
    x_axis = x_features
    if type(x_axis)==int:
        x_axis = range(1,x_axis+1)

    position = range(1,total + 1)
    fig = plt.figure(figsize=(16,15))

    for k in range(total):

        ax = fig.add_subplot(rows,cols,position[k])
        ax.title.set_text(keys[k-1])

        #ax.set_ylabel('Matrices Score')

        ax.plot(x_axis,result[keys[k-1]]['acc'],'teal',label='Acc',marker='o')
        ax.plot(x_axis,result[keys[k-1]]['prec'],'tomato', ls='dotted', label='Prec')
        ax.plot(x_axis,result[keys[k-1]]['rec'],'black', linestyle='--', label='Rec')
        ax.legend()
        ax.grid()

    fig.subplots_adjust(hspace=0.3)
    plt.show()




from sklearn.feature_selection import chi2,f_classif,mutual_info_classif

def apply_univeriate(X_in,Y_in,classifiers,k_list,selection_func=f_classif):
    models_acc={}
    
    for k in k_list:
        selected_x=SelectKBest(selection_func,k=k).fit_transform(X_in,Y_in)
        X_train,X_test,y_train,y_test = train_test_split(selected_x,Y_in,test_size=0.3)
    
        for model in classifiers:
            model_name=type(model).__name__
            if model_name not in models_acc:
                models_acc[model_name]={}
                models_acc[model_name]['acc']=[]
                models_acc[model_name]['prec']=[]
                models_acc[model_name]['rec']=[]
                
            fit = model.fit(X_train,y_train)
            y_predicted = fit.predict(X_test)
            acc_score= accuracy_score(y_test,y_predicted)
            models_acc[model_name]['acc'].append(acc_score)
            prec_score= precision_score(y_test,y_predicted)
            models_acc[model_name]['prec'].append(prec_score)
            rec_score= recall_score(y_test,y_predicted)
            models_acc[model_name]['rec'].append(rec_score)
    
    return(models_acc)



from statsmodels.stats.outliers_influence import variance_inflation_factor
def apply_vif(X_in,Y_in,classifiers):
    models_acc = {}
    num_features=[]
    vif = np.array([variance_inflation_factor(X_in.values,i) for i in range(X_in.shape[1])])
    print(sorted(vif))
    print('\nAverage: ',np.average(vif))
    thresh_list = [float(i) for i in(input('\nEnter Threshold List seperated by ,').split(','))]
    
    for thresh in thresh_list:
        selected_x = X_in.loc[:,vif<thresh]
        num_features.append(selected_x.shape[1])
        X_train,X_test,y_train,y_test = train_test_split(selected_x,Y_in,test_size=0.3)
        
        for model in classifiers:
            model_name=type(model).__name__
            if model_name not in models_acc:
                models_acc[model_name]={}
                models_acc[model_name]['acc']=[]
                models_acc[model_name]['prec']=[]
                models_acc[model_name]['rec']=[]
                
            fit = model.fit(X_train,y_train)
            y_predicted = fit.predict(X_test)
            acc_score= accuracy_score(y_test,y_predicted)
            models_acc[model_name]['acc'].append(acc_score)
            prec_score= precision_score(y_test,y_predicted)
            models_acc[model_name]['prec'].append(prec_score)
            rec_score= recall_score(y_test,y_predicted)
            models_acc[model_name]['rec'].append(rec_score)
    
    return(models_acc,thresh_list)
        

from sklearn.feature_selection import SelectFromModel
def apply_sel_from_model(selection_model,X_in,Y_in,classifiers,max_features_list):
    models_acc={}
    for model in classifiers:
        for k in max_features_list:
            mask = SelectFromModel(selection_model,max_features=k).fit(X,Y).get_support()
            selected_x = X_in.loc[:,mask]
            X_train,X_test,y_train,y_test = train_test_split(selected_x,Y_in,test_size=0.2)
            model_name=type(model).__name__
            if model_name not in models_acc:
                models_acc[model_name]={}
                models_acc[model_name]['acc']=[]
                models_acc[model_name]['prec']=[]
                models_acc[model_name]['rec']=[]
            
            fit = model.fit(X_train,y_train)
            y_predicted = fit.predict(X_test)
            acc_score= accuracy_score(y_test,y_predicted)
            models_acc[model_name]['acc'].append(acc_score)
            prec_score= precision_score(y_test,y_predicted)
            models_acc[model_name]['prec'].append(prec_score)
            rec_score= recall_score(y_test,y_predicted)
            models_acc[model_name]['rec'].append(rec_score)
    return (models_acc)


from sklearn.feature_selection import SelectKBest
def apply_Sel_K_Best_model(selection_method,X_in,Y_in,classifiers,max_features_list):
    models_acc={}
    for model in classifiers:
        
        for k in max_features_list:
            mask = SelectKBest(selection_method, k=k).fit(X,Y).get_support()
            selected_x = X_in.loc[:,mask]
            X_train,X_test,y_train,y_test = train_test_split(selected_x,Y_in,test_size=0.2)
            model_name=type(model).__name__
            
            if model_name not in models_acc:
                models_acc[model_name]={}
                models_acc[model_name]['acc']=[]
                models_acc[model_name]['prec']=[]
                models_acc[model_name]['rec']=[]
            
            fit = model.fit(X_train,y_train)
            y_predicted = fit.predict(X_test)
            acc_score= accuracy_score(y_test,y_predicted)
            models_acc[model_name]['acc'].append(acc_score)
            prec_score= precision_score(y_test,y_predicted)
            models_acc[model_name]['prec'].append(prec_score)
            rec_score= recall_score(y_test,y_predicted)
            models_acc[model_name]['rec'].append(rec_score)
            
    return (models_acc)


from sklearn.feature_selection import RFE
def apply_RFE(selection_method,X_in,Y_in,classifiers,max_features_list):
    models_acc={}
    for model in classifiers:
        
        for k in max_features_list:
            mask = RFE(selection_method, n_features_to_select=k).fit(X,Y).get_support()
            selected_x = X_in.loc[:,mask]
            X_train,X_test,y_train,y_test = train_test_split(selected_x,Y_in,test_size=0.2)
            model_name=type(model).__name__
            
            if model_name not in models_acc:
                models_acc[model_name]={}
                models_acc[model_name]['acc']=[]
                models_acc[model_name]['prec']=[]
                models_acc[model_name]['rec']=[]
            
            fit = model.fit(X_train,y_train)
            y_predicted = fit.predict(X_test)
            acc_score= accuracy_score(y_test,y_predicted)
            models_acc[model_name]['acc'].append(acc_score)
            prec_score= precision_score(y_test,y_predicted)
            models_acc[model_name]['prec'].append(prec_score)
            rec_score= recall_score(y_test,y_predicted)
            models_acc[model_name]['rec'].append(rec_score)
            
    return (models_acc)



from mlxtend.feature_selection import SequentialFeatureSelector
def apply_SFS(classifiers,X_in,Y_in,sel_feat='best'):
    models_result = {}
    models_result['Forward']=[]
    models_result['Backward']=[]
    
    for forward in [True,False]:
        for model in classifiers:
            model_name = type(model).__name__
            if model_name not in models_result:
                models_result[model_name] = {}
                models_result[model_name]['Forward Features'] = None
                models_result[model_name]['Forward Index'] = None
                models_result[model_name]['Backward Features'] = None
                models_result[model_name]['Backward Index'] = None
                
            sfs_obj = SequentialFeatureSelector(model,k_features=sel_feat, forward=forward)
            sfs = sfs_obj.fit(X_in,Y_in)
            
            
            if forward==True:
                models_result['Forward'].append(sfs.k_score_)
                models_result[model_name]['Forward Features'] = sfs.k_feature_names_
                models_result[model_name]['Forward Index'] = sfs.k_feature_idx_
            
            else:
                models_result['Backward'].append(sfs.k_score_)
                models_result[model_name]['Backward Features'] = sfs.k_feature_names_
                models_result[model_name]['Backward Index'] = sfs.k_feature_idx_
    
    return(models_result)


def plot_SFS_Forward_vs_Backward(result_dict, list_of_names_of_classifiers):
    plt.plot(classifiers_names,result['Forward'], marker='o',color='g', label='Forward')
    plt.plot(classifiers_names,result['Backward'], marker='*',color='r', label='Backward',ls='--')
    plt.grid()
    plt.xticks(rotation=75)
    plt.ylabel('Accuracy Score')
    plt.legend()
    plt.show()



from sklearn.decomposition import PCA
def apply_PCA(X_in,Y_in,classifiers,max_comp_list):
    models_acc={}
        
    for k in max_comp_list:
        selected_x = PCA(n_components=k).fit_transform(X_in)
        X_train,X_test,y_train,y_test = train_test_split(selected_x,Y_in,test_size=0.2)

        for model in classifiers:
            model_name=type(model).__name__
            
            if model_name not in models_acc:
                models_acc[model_name]={}
                models_acc[model_name]['acc']=[]
                models_acc[model_name]['prec']=[]
                models_acc[model_name]['rec']=[]
            
            fit = model.fit(X_train,y_train)
            y_predicted = fit.predict(X_test)
            acc_score= accuracy_score(y_test,y_predicted)
            models_acc[model_name]['acc'].append(acc_score)
            prec_score= precision_score(y_test,y_predicted)
            models_acc[model_name]['prec'].append(prec_score)
            rec_score= recall_score(y_test,y_predicted)
            models_acc[model_name]['rec'].append(rec_score)
            
    return (models_acc)





