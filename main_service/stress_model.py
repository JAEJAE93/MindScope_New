import pickle

import statistics
from collections import Counter
import datetime
import random

# Normalize
from sklearn.preprocessing import MinMaxScaler

# Model
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import classification_report, accuracy_score, f1_score

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

import shap
import pandas as pd
import numpy as np

from main_service.models import ModelResult
from main_service.models import AppUsed


## CASE 1 : Right after the end of step1
def makeLabel(user_stress_level_value_list):
    """
        Step1 끝난 시점에 한번만 계산 --> 스트레스 레벨을 세 구간으로 나눔
            - just call once after step1 (for calculating stress section)
        # bins = 데이터 범위를 동일한 길이로 N등분, ] = 닫혀있다, 포함한다.
        # example) 0 ~ 16일 경우: (-0.017 ~ 5.333], (5.333 ~ 10.667], (10.667 ~ 16]
        # https://rfriend.tistory.com/404
    """
    try:
        stress_level_series = pd.Series(user_stress_level_value_list)
        bin_stress_level = stress_level_series.value_counts(bins=3)
        bin_range_list = []

        for bin_index in bin_stress_level.index:  # type = interval
            left = bin_index.left
            right = bin_index.right
            bin_range_list.append(left)
            bin_range_list.append(right)

        bin_range_list = list(set(bin_range_list))
        bin_range_list.sort()

        stress_lv0_cut = bin_range_list[1]
        stress_lv1_cut = bin_range_list[2]
        stress_lv2_max = bin_range_list[3]

        return [stress_lv0_cut, stress_lv1_cut, stress_lv2_max]

    except Exception as e:
        print("Make label error: ", e)
        return [0, 0, 0]

class StressModel:
    # variable for setting label

    # variable for label
    CONST_STRESS_LOW = 0
    CONST_STRESS_LITTLE_HIGH = 1
    CONST_STRESS_HIGH = 2

    feature_df_with_state = pd.read_csv('assets/feature_with_state.csv')

    def __init__(self, uid, dayNo, emaNo, stress_lv0_max, stress_lv1_max, stress_lv2_min):
        self.uid = uid
        self.dayNo = dayNo
        self.emaNo = emaNo
        self.stress_lv0_max = stress_lv0_max
        self.stress_lv1_max = stress_lv1_max
        self.stress_lv2_min = stress_lv2_min

    def mapLabel(self, score):
        "값 수정할 것 0828"
        try:
            if score <= self.stress_lv0_max:
                return StressModel.CONST_STRESS_LOW

            elif (score > self.stress_lv0_max) and (score < self.stress_lv1_max):
                return StressModel.CONST_STRESS_LITTLE_HIGH

            elif (score >= self.stress_lv1_max):
                return StressModel.CONST_STRESS_HIGH
        except Exception as e:
            print(e)

    def preprocessing(self, df, prep_type):
        """
         - 1. del NAN or replace to zero
         - 2. mapping label (Stress lvl 0, 1, 2)
        """
        # print(".....preprocessing")

        delNan_col = ['Audio min.', 'Audio max.', 'Audio mean', 'Sleep dur.']
        try:
            for col in df.columns:
                if (col != 'Stress lvl') & (col != 'User id') & (col != 'Day'):
                    df[col] = df[col].replace('-', 0)

                    df[col] = pd.to_numeric(df[col])
                df = df.fillna(0)

            if prep_type == "default":
                df['Stress_label'] = df['Stress lvl'].apply(lambda score: self.mapLabel(score))
            else:
                df['Stress_label'] = -1
                df['Stress lvl'] = -1

        except Exception as e:
            print(e)

        # EMA order 중복 제거
        df = df.drop_duplicates(["Day", "EMA order"], keep="first")
        df = df.reset_index(drop=True)

        return df

    def normalizing(self, norm_type, preprocessed_df, new_row_preprocessed, user_email, day_num, ema_order):
        # print(".......normalizing")

        # user info columns
        userinfo = ['User id', 'Day', 'EMA order', 'Stress lvl', 'Stress_label']

        feature_scaled = pd.DataFrame()
        try:
            scaler = MinMaxScaler()

            feature_df = preprocessed_df[StressModel.feature_df_with_state['features'].values]
            uinfo_df = preprocessed_df[userinfo].reset_index(drop=True)

            if norm_type == "default":
                # feature list
                feature_scaled = pd.DataFrame(scaler.fit_transform(feature_df), columns=feature_df.columns)

            elif norm_type == "new":
                # 기존 data와 합치기
                feature_df = pd.concat([feature_df.reset_index(drop=True), new_row_preprocessed[
                    StressModel.feature_df_with_state['features'].values].reset_index(drop=True)])

                feature_scaled = pd.DataFrame(scaler.fit_transform(feature_df), columns=feature_df.columns)

                uinfo_df = uinfo_df.append({'User id': user_email, 'Day': day_num, 'EMA order': ema_order, 'Stress lvl': -1, 'Stress_label': -1}, ignore_index=True)

            feature_scaled = pd.concat([uinfo_df, feature_scaled.reset_index(drop=True)], axis=1)

        except Exception as e:
            print("feature scale func error: ", e)

        # EMA order 중복 제거
        feature_scaled = feature_scaled.drop_duplicates(["Day", "EMA order"], keep="first")
        feature_scaled = feature_scaled.reset_index(drop=True)

        return feature_scaled

    def initModel(self, norm_df):
        """
        initModel
        """
        print("stress_model.py... initModel...")

        try:
            print("Class Count... :", Counter(norm_df['Stress_label']))
            # *** 만약 훈련 데이터에 0,1,2 라벨 중 하나가 없다면? 하나의 라벨만 존재한다면?

            X = norm_df[StressModel.feature_df_with_state['features'].values]
            y = norm_df['Stress_label'].values
            # print("y len: ", len(np.unique(y)))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

            # XGBoost model
            model_clf = XGBClassifier(max_depth=7, n_estimators=100, seed=100)

            model_pred = model_clf.fit(X_train, y_train).predict(X_test)
            model_acc = accuracy_score(y_test, model_pred)
            model_f1 = f1_score(y_test, model_pred, average='weighted')

            model_result = {'acc' : model_acc,
                            'f1_score' : model_f1}

            # Random forest setting
            # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
            # model = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=100)
            #
            # # TODO split_num is 2 ==> related to stress prediction check(3)
            # split_num = 2
            # if len(Y) < split_num:
            #     split_num = len(Y_test)
            #
            # kfold = KFold(n_splits=split_num)
            # scoring = {'accuracy': 'accuracy',
            #            'f1_micro': 'f1_micro',
            #            'f1_macro': 'f1_macro'}
            #
            # cv_results = cross_validate(model, X_train, Y_train, cv=kfold, scoring=scoring)
            # model_result = {'accuracy': cv_results['test_accuracy'].mean(),
            #                 'f1_micro': cv_results['test_f1_micro'].mean(),
            #                 'f1_macro': cv_results['test_f1_macro'].mean()}
            # model.fit(X_train, Y_train)

            print("Model Result... : ", model_result)

            # TODO : Debugging 주석 처리
            with open('model_result/' + str(self.uid) + "_model.p", 'wb') as file:
                pickle.dump(model_clf, file)
                print("Model saved")

        except Exception as e:
            print("initModel func error: ", e)

    def saveAndGetSHAP(self, user_all_label, pred, new_row_raw, new_row_norm, initModel):
        start_time = datetime.datetime.now()
        model_results = []

        # 상위 버전 xgboost의 경우, 모델 encoding 버퍼 문제 발생
        # https://github.com/slundberg/shap/issues/1215

        xgb_booster = initModel.get_booster()
        model_bytearray = xgb_booster.save_raw()[4:]
        def byte_error(self=None):
            return model_bytearray

        xgb_booster.save_raw = byte_error

        features = StressModel.feature_df_with_state['features'].values
        feature_state_df = StressModel.feature_df_with_state
        model_accuracy = 0

        # model 성능 평가
        y_pred_proba = initModel.predict_proba(new_row_norm[features])
        model_accuracy = y_pred_proba[0]
        print("model_accuracy: ", model_accuracy)

        # shap setting
        shap.initjs()

        try:
            explainer = shap.TreeExplainer(xgb_booster)
        except Exception as e:
            print("shap tree explainer error: ", e)
        # explainer.feature_perturbation = "tree_path_dependent"

        shap_values = explainer.shap_values(new_row_norm[features])
        # print(shap_values)
        expected_value = explainer.expected_value

        # ## TODO : SHAP Exception 발생 가능 부분 ==> SHAP 에서 적은 빈도수의 Label 해석 안줄때/...혹시나 해서 모델 한번 더 학습
        # try:
        #     print("expected_value: ", type(expected_value))
        #     print("expected_value: ", expected_value.shape[0])
        #     if (expected_value.shape[0]) != len(user_all_label):
        #         print("Shap if statement...")
        #         with open('data_result/' + str(self.uid) + "_features.p", 'rb') as file:
        #             preprocessed = pickle.load(file)
        #
        #         norm_df = StressModel.normalizing(self, "default", preprocessed, None, None, None, None)
        #         StressModel.initModel(self, norm_df)
        #
        #         explainer = shap.TreeExplainer(initModel)
        #         explainer.feature_perturbation = "tree_path_dependent"
        #
        #         features = StressModel.feature_df_with_state['features'].values
        #         feature_state_df = StressModel.feature_df_with_state
        #
        #         ###  model 성능 평가
        #         y_pred_proba = initModel.predict_proba(new_row_norm[features])
        #         model_accuracy = y_pred_proba[0]
        #         print("if model_accuracy: ", model_accuracy)
        #
        #         shap_values = explainer.shap_values(new_row_norm[features])
        #         # print(shap_values)
        #         expected_value = explainer.expected_value
        #         # print("len expected_value: ", len(expected_value))
        # except Exception as e:
        #     print("SHAP label length error: ", e)
        #     pass


        check_label = [0 for i in range(3)]
        # not_user_label_list = list(set(check_label) - set(user_all_label)) # 유저한테 없는 label 계산

        try:
            for label in user_all_label:  # 유저한테 있는 Stress label 에 따라
                feature_list = ""

                index = user_all_label.index(label)
                # shap_accuracy = expected_value[index]
                shap_list = shap_values[index]

                if len(shap_list.shape) == 1: ## EXCEPTION CASE..
                    shap_dict = dict(zip(features, shap_list))
                else:
                    shap_dict = dict(zip(features, shap_list[0]))

                shap_dict_sorted = sorted(shap_dict.items(), key=(lambda x: x[1]), reverse=True)

                # act_features = ['Duration WALKING', 'Duration RUNNING', 'Duration BICYCLE', 'Duration ON_FOOT', 'Duration VEHICLE']
                app_features = ['Social & Communication','Entertainment & Music','Utilities','Shopping', 'Games & Comics',
                                'Health & Wellness', 'Education', 'Travel', 'Art & Design & Photo', 'News & Magazine', 'Food & Drink']

                act_tmp = ""
                for feature_name, s_value in shap_dict_sorted:
                    if s_value > 0: #check
                        feature_id = feature_state_df[feature_state_df['features'] == feature_name]['feature_id'].values[0]
                        feature_value = new_row_norm[feature_name].values[0]
                        ## TODO : 데이터가 전부 다 0인 경우..추가 작업이 필요할 수 있음
                        # 현재는 feature_list가 0일 경우, NO_FEATURES 반환
                        if new_row_raw[feature_name].values[0] != 0:
                            # ACT FEATURE
                            # if feature_name in act_features:
                            #     if act_tmp == "":
                            #         act_tmp += feature_name
                            #
                            #         if feature_value >= 0.5:
                            #             feature_list += str(feature_id) + '-high '
                            #         else:
                            #             feature_list += str(feature_id) + '-low '
                            if feature_name in app_features:
                                # Add package
                                try:
                                    pkg_result = AppUsed.objects.get(uid=self.uid, day_num=self.dayNo, ema_order=self.emaNo)
                                    pkg_text = ""
                                    if feature_name == "Entertainment & Music":
                                        pkg_text = pkg_result.Entertainment_Music
                                    elif feature_name == "Utilities":
                                        pkg_text = pkg_result.Utilities
                                    elif feature_name == "Shopping":
                                        pkg_text = pkg_result.Shopping
                                    elif feature_name == "Games & Comics":
                                        pkg_text = pkg_result.Games_Comics
                                    elif feature_name == "Others":
                                        pkg_text = pkg_result.Others
                                    elif feature_name == "Health & Wellness":
                                        pkg_text = pkg_result.Health_Wellness
                                    elif feature_name == "Social & Communication":
                                        pkg_text = pkg_result.Social_Communication
                                    elif feature_name == "Education":
                                        pkg_text = pkg_result.Education
                                    elif feature_name == "Travel":
                                        pkg_text = pkg_result.Travel
                                    elif feature_name == "Art & Design & Photo":
                                        pkg_text = pkg_result.Art_Photo
                                    elif feature_name == "News & Magazine":
                                        pkg_text = pkg_result.News_Magazine
                                    elif feature_name == "Food & Drink":
                                        pkg_text = pkg_result.Food_Drink

                                    if pkg_text != "":
                                        if feature_value >= 0.5:
                                            feature_list += str(feature_id) + '-high&' + pkg_text + " "
                                        else:
                                            feature_list += str(feature_id) + '-low '

                                except Exception as e:
                                    print("Exception during making feature_list of app...get AppUsed db", e)

                            else:
                                if feature_value >= 0.5:
                                    feature_list += str(feature_id) + '-high '
                                else:
                                    feature_list += str(feature_id) + '-low '

                if feature_list == "":
                    feature_list = "NO_FEATURES"

                try:
                    if label == pred:
                        model_result = ModelResult.objects.create(uid=self.uid, timestamp=start_time,
                                                                  day_num=self.dayNo, ema_order=self.emaNo,
                                                                  prediction_result=label, accuracy=model_accuracy[label],
                                                                  feature_ids=feature_list, model_tag=True)
                    else:
                        model_result = ModelResult.objects.create(uid=self.uid, timestamp=start_time,
                                                                  day_num=self.dayNo, ema_order=self.emaNo,
                                                                  prediction_result=label, accuracy=model_accuracy[label],
                                                                  feature_ids=feature_list)
                except Exception as e:
                    print("ModelResult.objects.create error: ", e)

                check_label[label] = 1
                model_results.append(model_result)

        except Exception as e:
            print("Exception at saveAndGetSHAP error: ", e)
            pass

        try:
            ## For 문 끝난 후, model_result 에 없는 stress lvl 추가 & 일반적인 문구 추가
            for i in range(3):
                if check_label[i] == 0:
                    # random_acc = random.uniform(0.0, 1.0)
                    # random_acc = round(random_acc, 2)
                    try:
                        if i == 0 : # LOW General message, 마지막 띄어쓰기 조심!
                            feature_list = '0-general_0 7-general_0 11-general_0 17-general_0 28-general_0 '
                            model_result = ModelResult.objects.create(uid=self.uid, timestamp=start_time,
                                                                      day_num=self.dayNo, ema_order=self.emaNo,
                                                                      prediction_result=i, accuracy=0,
                                                                      feature_ids=feature_list)
                        else: #LITTLE HIGH, HIGH General message
                            feature_list = '0-general_1 7-general_1 11-general_1 17-general_1 28-general_1 '
                            model_result = ModelResult.objects.create(uid=self.uid, timestamp=start_time,
                                                                      day_num=self.dayNo, ema_order=self.emaNo,
                                                                      prediction_result=i, accuracy=0,
                                                                      feature_ids=feature_list)
                    except Exception as e:
                        print("model result에 없는 stress lvl 추가 오류: ", e)

                    model_results.append(model_result)

        except Exception as e:
            print("saveAndGetSHAP general statement error: ",e)

        # print("Total SaveAndGetSHAP Working... ", datetime.datetime.now() - start_time) # 시간 1초도 안 걸림

        return model_results

    # def update(self, user_response, day_num, ema_order):
    def update(self, user_response, day_num, ema_order, user_tag):
        start_update_time = datetime.datetime.now()
        print("Start Update func...")
        try:
            # update dataframe
            with open('data_result/' + str(self.uid) + "_features.p", 'rb') as file:
                preprocessed = pickle.load(file)

            # TODO : iloc 이나 다른 방법 사용해보기
            preprocessed.loc[(preprocessed['Day'] == day_num) & (preprocessed['EMA order'] == ema_order), 'Stress_label'] = user_response

            if user_tag == False:
                # save dataframe & retrain model
                try:
                    with open('data_result/' + str(self.uid) + "_features.p", 'wb') as file:
                        pickle.dump(preprocessed, file)
                    print("User_tag False update dataframe...")
                except Exception as e:
                    print("User_tag False error...: ", e)
                    pass
                # retrain the model
                try:
                    norm_df = StressModel.normalizing(self, "default", preprocessed, None, None, None, None)
                    StressModel.initModel(self, norm_df)

                    # update ModelResult Table
                    model_result = ModelResult.objects.get(uid=self.uid, day_num=day_num, ema_order=ema_order,
                                                           prediction_result=user_response)
                    model_result.user_tag = True
                    model_result.save()
                    print("Update model...")

                except Exception as e:
                    print("Retrain error: ", e)
                    pass
            else:
                # save dataframe another space
                # 지속적인 관찰을 위하여 다른 폴더에 데이터를 저장
                try:
                    with open('C:\\Users\\USER\\Desktop\\JH\\all_update_data\\' + str(self.uid) + "_features.p", 'wb') as file:
                        pickle.dump(preprocessed, file)
                    print("User_tag True update dataframe...")

                except Exception as e:
                    print("user_tag True error...: ", e)
                    pass

        except Exception as e:
            print("Update total error: ", e)

        print("Total update time... ", datetime.datetime.now() - start_update_time)