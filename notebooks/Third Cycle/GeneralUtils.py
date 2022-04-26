import numpy   as np
import pandas  as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats as ss

from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.calibration import calibration_curve
from scikitplot.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, cohen_kappa_score, f1_score, roc_curve, roc_auc_score

class Utils():
    
    def __init__(self):
        self.pallet = sns.diverging_palette(359, 359, n=5, s=999, l=50, center='dark')
    
    def test(self):
        print("Olá Mundo")
    
    def min_max_all_features(self, df, mms=False):
        if mms == False:
            mms = MinMaxScaler()
        
        df['bmi'] = mms.fit_transform(df[['bmi']].values)
        df['age'] = mms.fit_transform(df[['age']].values)
        df['gluc'] = mms.fit_transform(df[['gluc']].values)
        df['alco'] = mms.fit_transform(df[['alco']].values)
        df['smoke'] = mms.fit_transform(df[['smoke']].values)
        df['active'] = mms.fit_transform(df[['active']].values)
        df['gender'] = mms.fit_transform(df[['gender']].values)
        df['height'] = mms.fit_transform(df[['height']].values)
        df['weight'] = mms.fit_transform(df[['weight']].values)
        df['systolic'] = mms.fit_transform(df[['systolic']].values)
        df['diastolic'] = mms.fit_transform(df[['diastolic']].values)
        df['cholesterol'] = mms.fit_transform(df[['cholesterol']].values)
        df['pulse_pressure'] = mms.fit_transform(df[['pulse_pressure']].values)

        return df

    def standart_all_features(self, df, sss=False):
        if sss == False:
            sss = StandardScaler()
            
        df['bmi'] = sss.fit_transform(df[['bmi']].values)
        df['age'] = sss.fit_transform(df[['age']].values)
        df['gluc'] = sss.fit_transform(df[['gluc']].values)
        df['alco'] = sss.fit_transform(df[['alco']].values)
        df['smoke'] = sss.fit_transform(df[['smoke']].values)
        df['active'] = sss.fit_transform(df[['active']].values)
        df['gender'] = sss.fit_transform(df[['gender']].values)
        df['height'] = sss.fit_transform(df[['height']].values)
        df['weight'] = sss.fit_transform(df[['weight']].values)
        df['systolic'] = sss.fit_transform(df[['systolic']].values)
        df['diastolic'] = sss.fit_transform(df[['diastolic']].values)
        df['cholesterol'] = sss.fit_transform(df[['cholesterol']].values)
        df['pulse_pressure'] = sss.fit_transform(df[['pulse_pressure']].values)

        return df
    
    def simple_metrics(self, df, proportion=.2, with_trim=True, num=False):
        num_att = df.select_dtypes(include=['int64', 'float64'])
        if num:
            return num_att
        else:
            c1 = pd.DataFrame(num_att.apply(np.mean)).T
            c2 = pd.DataFrame(num_att.apply(np.median)).T
            d1 = pd.DataFrame(num_att.apply(min)).T
            d2 = pd.DataFrame(num_att.apply(max)).T
            d3 = pd.DataFrame(num_att.apply(np.std)).T
            d4 = pd.DataFrame(num_att.apply(lambda x: x.max() - x.min())).T
            d5 = pd.DataFrame(num_att.apply(lambda x: x.skew())).T
            d6 = pd.DataFrame(num_att.apply(lambda x: x.kurtosis())).T

            m = pd.concat([d1, d2, c1, c2, d3, d4, d5, d6], axis=0).T.reset_index()

            if with_trim:
                trim_mean = pd.DataFrame(ss.trim_mean(num_att.values, proportion))
                m = pd.concat([m, trim_mean[0]], axis=1)
                m.columns = ['att', 'min', 'max', 'mean', 'median', 'std', 'range', 'skew', 'kurtosis', 'trim_mean']
                m = m[['att', 'min', 'max', 'mean', 'trim_mean', 'median', 'std', 'range', 'skew', 'kurtosis']]

                return m

            else:
                m.columns = ['att', 'min', 'max', 'mean', 'median', 'std', 'range', 'skew', 'kurtosis']
                m = m[['att', 'min', 'max', 'mean', 'median', 'std', 'range', 'skew', 'kurtosis']]

                return m

    def args(self, bins=np.arange(0, 2, 1), hstep='step', lwidth=3, c='r', label='Sales', normed=False):
        return {'bins': bins, 'histtype': hstep, 'linewidth': lwidth, 'color': c, 'density': normed, 'label': label}

    def args_b(self, color=(1, 1, 1, 0), edgecolor='red', linewidth=3):
        return {'linewidth': linewidth, 'color': color, 'edgecolor': edgecolor }

    def plot_count_gender(self, df, aux):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        sns.countplot(df['cardio'], ax=ax[0], **self.args_b());
        sns.barplot(aux['cardio'], aux['gender'], **self.args_b(edgecolor='b'))
        ax[0].set_title('Count per Class');
        ax[1].set_title('Count per Gender');

        return None

    def plot_cardio(self, aux1, aux2, aux3, label):
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

        ax1.bar(aux2[label], aux2['cardio'], **self.args_b(linewidth=2, edgecolor='b'))
        ax1.set_title('With Cardio')
        if label == 'age':
            ax1.vlines(45, 0, aux2['cardio'].max(), 'r--')
            ax1.hlines(850, 40, aux2[label].max(), 'r--')
        ax2.bar(aux3[label], aux3['cardio'], **self.args_b(linewidth=2, edgecolor='b'))
        ax2.set_title('Without Cardio')
        sns.barplot(aux1[label], aux1['cardio'], palette=self.palette, ax=ax3);
        ax3.set_title('Full Dataset');

        return None

    def plot_pulse_pression(self, aux1, aux2, aux3):
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

        ax1.bar(aux2['age'], aux2['pulse_pressure'], **self.args_b(edgecolor='b'))
        ax1.set_title('With Cardio')
        ax2.bar(aux3['age'], aux3['pulse_pressure'], **self.args_b(edgecolor='b'))
        ax2.set_title('Without Cardio')
        sns.barplot(aux1['age'], aux1['pulse_pressure'], palette=self.palette, ax=ax3)

        return None

    def plot_gluc(self, aux1, aux2, aux3, aux4, aux5):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].pie(aux1['cardio'], labels=[1, 2, 3], explode=[.14, .1, .1], colors=['k', 'b', 'r']);
        ax[0].set_title('Gluc Level');
        ax[0].legend();

        for i in zip([aux2, aux3], [aux4, aux5], ['diastolic']*2, ['systolic']*2, ['r', 'b'] ):
            ax[1].bar(i[0]['gluc'], i[0][i[2]], **self.args_b(edgecolor=i[4]))
            ax[2].bar(i[1]['gluc'], i[1][i[3]], **self.args_b(edgecolor=i[4]))

        for k in zip(range(1, 3), ['Gluc Level']*2, ['Diastolic', 'Systolic']):
            ax[k[0]].legend(['With Cardio', 'Without Cardio'])
            ax[k[0]].set_xlabel(k[1])
            ax[k[0]].set_ylabel(k[2])

        return None

    def plot_hist(self, df, ranges, msk):
        fig, ax = plt.subplots(ranges[0], ranges[1], figsize=(20, 30))
        ax = ax.flatten()
        for c, i in zip(df.columns.tolist(), range(len(df.columns))):
            ax[i].hist(df[c][msk], **self.args(bins=10, c='b', lwidth=3, label="Cardio") );
            ax[i].hist(df[c][~msk], **self.args(bins=10, lwidth=3, label="No Cardio"));
            ax[i].set_title(c)
            ax[i].legend()

        return None

    def qq_plot(self, df, ranges):
        fig, ax = plt.subplots(ranges[0], ranges[1], figsize=(15, 25))
        ax = ax.flatten()
        for c, i in zip(df.columns.tolist(), range(len(df.columns))):
            ss.probplot(df[c], plot=ax[i]);
            plt.tight_layout(w_pad=2., h_pad=2.)
            ax[i].text(.1, .9, c, color='maroon', transform=ax[i].transAxes)

        return None

    def get_importances(self, model, columns):
        aux = {}
        for k in range(len(columns)):
            aux[columns[k]] = model.feature_importances_[k]

        return aux

    def plot_importances(self, model1, model2, columns):
        model1_feat = self.get_importances(model1, columns)
        model2_feat = self.get_importances(model2, columns)

        fig, ax = plt.subplots(2, 1, figsize=(15, 10))
        ax[0].bar(model1_feat.keys(), model1_feat.values(), **self.args_b(edgecolor='k'))
        ax[1].bar(model2_feat.keys(), model2_feat.values(), **self.args_b(edgecolor='b'))
        ax[0].set_title('Random Forest Features')
        ax[1].set_title('XGBoost Features');

        return None

    def cumulative_gain(self, y_true, y_score, l):
        y_true = (y_true == l)

        index = np.argsort(y_score)
        y_true = y_true[index]

        gains = np.cumsum(y_true) / float(np.sum(y_true)) 
        percentages = np.arange(start=1, stop=len(y_true) + 1) / float(len(y_true))

        return percentages, gains

    def plot_cumulative(self, y_true, y_probas, title='Cumulative Curve', ax=None):
        percentages, gains1 = self.cumulative_gain(y_true, y_probas[:, 0], 1)
        percentages, gains2 = self.cumulative_gain(y_true, y_probas[:, 1], 0)

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))

        ax.set_title(title)

        for k in zip([percentages, percentages], [gains1, gains2], ['b', 'r'], ['Class 0', 'Class 1']):
            ax.plot(k[0], k[1], lw=3, color=k[2], label=k[3])

        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])

        ax.set_xlabel('Sample Size')
        ax.set_ylabel('Proportion Cardio')
        ax.grid('on')
        ax.legend(loc='lower right')

        return ax

    def plot_roc(self, y_true, yhat_p, title="Roc Curve", ax=None, model_name="baseline", test=True):
        fpr, tpr, _ = roc_curve(y_true, yhat_p[:,1])

        if ax == None:
            fig, ax = plt.subplots(figsize=(5, 5))

        ax.set_title(title)

        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.plot(fpr, tpr, 'b', lw=3, label=f'Model {model_name}')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.grid('on')
        ax.legend(loc="lower right")

        return ax
    
    def plot_3d_components(df_pca, title="3D PCA Components"):
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(df_pca[0], df_pca[1], df_pca[2])
        ax.set_title(title)
        ax.set_xlabel("0", color="b")
        ax.set_ylabel("1")
        ax.set_zlabel("2", color="r");

    def ml_error(self, model_name, y_test, yhat, yhat_p=False, roc=False):
        acc = accuracy_score(y_test, yhat)
        rec = recall_score(y_test, yhat)
        pre = precision_score(y_test, yhat)
        fsc = f1_score(y_test, yhat)
        kpp = cohen_kappa_score(y_test, yhat)
        rau = roc_auc_score(y_test, yhat) 

        if roc:
            fig, ax = plt.subplots(1, 3, figsize=(15,5))
            self.plot_cumulative(y_test, yhat_p, ax=ax[0])
            self.plot_roc(y_test, yhat_p, ax=ax[1], model_name=model_name)
            plot_confusion_matrix(y_test, yhat, ax=ax[2])

        else:
             plot_confusion_matrix(y_test, yhat, figsize=(15, 5))

        return pd.DataFrame({'Model Name': model_name,
                               'Accuracy': acc,
                               'Precision': pre,
                               'Kappa Score': kpp,
                               'ROCAUC': rau,
                               'F1-Score':fsc}, index=[0])

    def cross_validation(self, model, x, y, k, verb, test=False, df_test=False):
        kfold = StratifiedKFold(n_splits=k, shuffle=True)

        i=1
        pre_list = []
        acc_list = []

        for train_ix, val_ix in kfold.split(x, y):
            if verb:
                print(f'{i} -> {k} Cross Validation Folds')

            # Get Dataset Folders
            x_train_fold = x.iloc[train_ix]
            y_train_fold = y.iloc[train_ix]
            x_val_fold = x.iloc[val_ix]
            y_val_fold = y.iloc[val_ix]

            if test:
                x_test = df_test.iloc[:,:-1]
                y_test = df_test.iloc[:, -1]

                # Model Train 
                model = model.fit(x_train_fold, y_train_fold)
                yhat_label = model.predict(x_test)
                #yhat_proba = model.predict_proba(x_val_fold)

                acc_list.append(accuracy_score(y_test, yhat_label))
                pre_list.append(precision_score(y_test, yhat_label))

            else:
                # Model Train 
                model = model.fit(x_train_fold, y_train_fold)
                yhat_label = model.predict(x_val_fold)
                #yhat_proba = model.predict_proba(x_val_fold)

                acc_list.append(accuracy_score(y_val_fold, yhat_label))
                pre_list.append(precision_score(y_val_fold, yhat_label))


            i+=1

        df = pd.DataFrame(columns=["Model Name", "Accuracy", "Precision"], index=[-1])
        df["Model Name"] = df["Model Name"].fillna(type(model).__name__)
        df["Accuracy"]  = str(round(np.mean(acc_list), 4)) + " + / - " + str(round(np.std(acc_list), 4)) 
        df["Precision"] = str(round(np.mean(pre_list), 4)) + " + / - " + str(round(np.std(pre_list), 4))

        return df.reset_index(drop=True)


    def perform_bootstrap(self, n_inter, df, model, x_test, y_test, verb=False):
        stats = []
        i = 0

        for i in range(n_inter):
            if verb:
                print(f'Iter: {i} | {n_inter}')
            train = resample( df.values, n_samples=int(len(df) * .30))
            #test = np.array([x for x in data if x.tolist() not in train.tolist()])

            model.fit(train[:, :-1], train[:, -1])
            yhat = model.predict(x_test.values)

            acc = precision_score( y_test, yhat )
            stats.append( acc )
            i+=1

        return stats

    def pair_plot_calibration_curve(self, model, y_test, yhat_probas, yhat_cali_probas):
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot([1,0], [1,0], 'k--', label="Perfect Calibration")

        probas = [k[:, 1] for k in [yhat_probas, yhat_cali_probas]]
        for i, l in zip(probas, [' No Calibrated', ' Calibrated'] ):
            prob_true, prob_pred  = calibration_curve(y_test, i, n_bins=10, normalize=True)
            ax.plot(prob_pred, prob_true, 's-', label=type(model).__name__ + l);

        ax.set_title("Calibration Curves")
        ax.set_xlabel("Positive Fractions")
        ax.set_ylabel("Avg Probas")
        plt.legend();

    def plot_bootstrap_iters(self, stats, n_inter=1000):
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        c, _, _, = ax[0].hist(stats, **self.args(bins=25, lwidth=2, c='b'));
        ax[0].set_title(f'Bootstrap for {n_inter} Iter');

        for i in zip( [.25, .50, .75], [.8, .6, .4], ['k', 'r', 'orange'] ):
            a = i[0]
            p_l = ((1.-a)/2.) * 100
            p_u = (a+((1.-a)/2.)) * 100
            lower = np.percentile(stats, p_l)
            upper = np.percentile(stats, p_u)

            ax[0].vlines(lower, 0, max(c), color=i[2], linestyle="--")
            ax[0].vlines(upper, 0, max(c), color=i[2], linestyle="--")

            text2 = f'{int(a*100)}% confidence interval of Model Performace | {np.round(lower*100, 2)}% | and | {np.round(upper*100, 2)}% |'
            ax[1].text( .05, i[1], text2, bbox={'facecolor': 'white'}, transform=ax[1].transAxes, color='k');


    def prepare_dataset(self, df, mms, rs):
        df = df.rename(columns={'cardio.1': 'cardio'})
        df['gender'] = df['gender'].apply( lambda x: 1 if x == 2 else 0 )

        gluc_frequency = df.gluc.value_counts() / len(df)
        cholesterol_frequency = df.cholesterol.value_counts() / len(df)
        df.gluc = df.gluc.apply( lambda x: gluc_frequency[x] )
        df.cholesterol = df.cholesterol.apply( lambda x: cholesterol_frequency[x] )

        df['age'] = mms.fit_transform(df[['age']].values)
        df['height'] = mms.fit_transform(df[['height']].values)
        df['systolic'] = mms.fit_transform(df[['systolic']].values)
        df['diastolic'] = mms.fit_transform(df[['diastolic']].values)

        df['bmi'] = rs.fit_transform(df[['bmi']].values)
        df['pulse_pressure'] = rs.fit_transform(df[['pulse_pressure']].values)

        return df
   