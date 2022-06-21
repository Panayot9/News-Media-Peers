
# load libraries
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


# Train sizes : epoch value and will be range(1,101)
# Training scores : cross validation scores. value is 100*1 (xxx/3873 values)
class LearningCurve:
    
    def smooth(self, scalars, weight: float):  # Weight between 0 and 1
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)                        # Save it
            last = smoothed_val                                  # Anchor the last smoothed value
            
        return smoothed

    def draw(self, train_sizes, train_scores, label="Training Score", yLabel="Accuracy Score") :

        if train_sizes is None:
            train_sizes=np.linspace(0.01, 1.0, 99)
        # Create means and standard deviations of training set scores
        # train_mean = np.mean(train_scores, axis=0)
        # train_std = np.std(train_scores, axis=0)

        train_scores = self.smooth(train_scores, 0.5)

        # Draw lines
        plt.subplots(1, figsize=(10,10))
        # plt.plot(train_sizes,'--', color="#111111")
        plt.plot(train_scores, color="#111111",  label=label)


        # Draw bands
        # plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")

        # Create plot
        plt.title("Learning Curve")
        plt.xlabel("Epoch"), plt.ylabel(yLabel), plt.legend(loc="best")
        plt.tight_layout(); plt.show()


    def draw_roc_curve(y_test, y_pred) :

        # roc curve for models
        fpr1, tpr1, thresh1 = roc_curve(y_test, y_pred, pos_label=1)
        fpr2, tpr2, thresh2 = roc_curve(y_test, y_pred, pos_label=1)
        # fpr1, tpr1, thresh1 = roc_curve(y_test, y_pred[:,1], pos_label=1)
        # fpr2, tpr2, thresh2 = roc_curve(y_test, y_pred[:,1], pos_label=1)

        # roc curve for tpr = fpr 
        random_probs = [0 for i in range(len(y_test))]
        p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

        print(p_fpr, p_tpr)


    def gen_roc(result, test_name) :
        th = [1, 0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.59,
      0.58, 0.57, 0.56, 0.55, 0.54, 0.53, 0.52, 0.51, 0.5, 0.499,
      0.498, 0.497, 0.496, 0.495, 0.494, 0.493, 0.492, 0.491, 0.49,
      0.47, 0.45, 0.43, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0]

#         print(len(result))
#         print(result)
        scores = []
        for t in th:
            tp = len(result[(result['mal']>=t) & (result['y'] == 1)])
            tn = len(result[(result['mal']<t) & (result['y'] == 0)])
            fp = len(result[(result['mal']>=t) & (result['y'] == 0)])
            fn = len(result[(result['mal']<t) & (result['y'] == 1)])
                    
            scores.append([t, float(fp)/(fp + tn),float(tp)/(tp + fn)])
        
#             print(t, float((tp+tn)/(tp+tn+fp+fn)))
        df_scores = pd.DataFrame().from_records(scores)
        df_scores.columns = ['threshold', 'fpr', 'tpr']
        df_scores.head(5)

        plt.plot(df_scores['fpr'], df_scores['tpr'])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.rcParams['font.size'] = 12
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.grid(True)
        plt.show()
        
        if os.path.exists('./data/bp_results/'):
            df_scores.to_csv(f'./data/bp_results/{test_name}.csv', index=False)

    def save_prob_result(test_name, result, y, test_mask, nodes=None, raw_directory='./data/raw_results/'):
        df_result = pd.DataFrame()
        df_result['node'] = nodes
        df_result['ben'] = result.t().tolist()[0]
        df_result['mal'] = result.t().tolist()[1]
        df_result['y'] = y.cpu()
        df_result.to_csv(f'{raw_directory}{test_name}.csv', index=False)
        
#         .data[data_test.test_mask]
        LearningCurve.gen_roc(df_result.loc[test_mask], test_name)
        
    def get_roc_files(path, training_percentage):
        datafiles = []
        for i in range(5):
            datafiles.append(path + f'/{i}_{training_percentage}.csv')
        return datafiles

    def get_avg_roc(datafiles):
        df_master = None
        isFirst = True
        for datafile in datafiles:
            df = pd.read_csv(datafile)
            if isFirst:
                df_master = df
                isFirst = False
            else:
                for col in df.columns:
                    df_master[col] = df_master[col] + df[col]

        print(df_master)
        for col in df_master.columns:
            df_master[col] = df_master[col] / len(datafiles)

        print(df_master)
        return df_master

    def plot_rocs(dfs, labels):
        i = 0
        for df in dfs:
            plt.plot(df['fpr'], df['tpr'], label = labels[i])
            i += 1
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.rcParams['font.size'] = 12
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        plt.grid(True)
        plt.show()

    def generate_roc_curves(filenames, base_path='./data/bp_results'):
        
        dataframes = []
        for filename in filenames:
            datafiles = LearningCurve.get_roc_files(base_path, filename)
            df = LearningCurve.get_avg_roc(datafiles)
            dataframes.append(df)

        LearningCurve.plot_rocs(dataframes, 
            filenames)
        