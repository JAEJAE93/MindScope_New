import json
import pickle
import sys
from sys import getsizeof

from main_service.grpc_handler import GrpcHandler
from main_service.models import ModelResult
from main_service.feature_extraction import Features

import threading
import schedule
import time
import pandas as pd
import datetime

from main_service.stress_model import StressModel, makeLabel

# region Notes:
#################################################################################################
# This is the Mindscope Server application
# It runs a prediction_task() function at every prediction time (prediction_times)
# Inside prediction task we have following steps for each user in the study:
# Step1. Check if the users day num is 14 and job is for initial model training
# Step2. Check if users day num is more than 14 days, only then extract features and make prediction
# Step3. Retrieve the last 4 hours data from the gRPC server
# Step4. Extract features from retrieved data
# Step5. Pre-process and normalize the extracted features
# Step6. Get trained stress prediction model
# Step7. Make prediction using current extracted features
# Step8. Insert a new pre-processed feature entry together with it's predicted label in DB for further model re-train
# Step9. Save prediction and important features in DB
# Step10. Construct a result message and send it to gRPC server with "STRESS_PREDICTION" data source id
# Step11. Lastly, check if user self reported his stress, then update the DB of pre-processed features with reported stress label
#################################################################################################
# endregion

# region Constants
FLAG_EMA_ORDER_1 = 0
FLAG_EMA_ORDER_2 = 1
FLAG_EMA_ORDER_3 = 2
FLAG_EMA_ORDER_4 = 3
FLAG_INITIAL_MODEL_TRAIN = -1

PREDICTION_HOURS = [11, 15, 19, 23]
SURVEY_DURATION = 10

# in days  ####### 변경 될 부분
SERVER_IP_PORT = '165.246.21.202:50051'
CAMPAIGN_ID = 23 ####### 변결될 부분

MANAGER_ID = 21
MANAGER_EMAIL = 'mindscope.nsl@gmail.com'
# endregion444444444444444444444444444444

# region Variables
run_service = None
thread = None
grpc_handler = None

# endregion

### test
def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

######

def start():
    global run_service, thread
    run_service = True
    thread = threading.Thread(target=service_routine())
    thread.start()

def stop():
    global run_service
    run_service = False

def service_routine():
    print("Run server...")
    # TODO : Debugging 시에 주석처리
    # prediction_task(FLAG_INITIAL_MODEL_TRAIN) ## for test

    # TODO : Debugging 시에 주석처리
    job_regular_at_11 = schedule.every().day.at("10:30").do(prediction_task, FLAG_EMA_ORDER_1)
    job_regular_at_15 = schedule.every().day.at("14:30").do(prediction_task, FLAG_EMA_ORDER_2)
    job_regular_at_19 = schedule.every().day.at("18:30").do(prediction_task, FLAG_EMA_ORDER_3)
    job_regular_at_23 = schedule.every().day.at("22:30").do(prediction_task, FLAG_EMA_ORDER_4)

    # TODO  : SURVEY_DURATION-1 일차, 밤 11 30분에 모델 초기화 / 0831 잠시 멈춤 -> 전체 데이터 진행해야 됨.
    job_initial_train = schedule.every().day.at("23:15").do(prediction_task, FLAG_INITIAL_MODEL_TRAIN) # 23:15

    while run_service:
       try:
           schedule.run_pending()
           time.sleep(1)
       except KeyboardInterrupt:
           stop()

    schedule.cancel_job(job=job_regular_at_11)
    schedule.cancel_job(job=job_regular_at_15)
    schedule.cancel_job(job=job_regular_at_19)
    schedule.cancel_job(job=job_regular_at_23)
    schedule.cancel_job(job=job_initial_train) # 마지막날 11시

    exit(0)

def prediction_task(i):
    start_prediction_task = datetime.datetime.now()
    print("#" * 30)
    print("\nStart prediction_task...", start_prediction_task)

    global grpc_handler

    if i == -1:
        print("Initial model training task is running...")
    else:
        print("Regular task for hour {} is running...".format(PREDICTION_HOURS[i]))

    grpc_handler = GrpcHandler(SERVER_IP_PORT, MANAGER_ID, MANAGER_EMAIL, CAMPAIGN_ID)

    now_time = int(datetime.datetime.now().timestamp()) * 1000
    from_time = now_time - (4 * 3600 * 1000)  # from 4 hours before now time

    users_info = grpc_handler.grpc_load_user_emails()
    ema_order = i + 1

    data_sources = {}
    campaign_start_time = 0
    campaign_info = grpc_handler.grpc_get_campaign_info()

    if campaign_info:
        for data_source in json.loads(campaign_info.configJson):
            data_sources[data_source['name']] = data_source['data_source_id']

        campaign_start_time = int(campaign_info.startTimestamp)

    else:
        print("No Campaign Info")
        exit(1)

    # check if the study duration day is more one day to the SURVEY_DURATION day then do following
    # get SURVEY_EMA data of all participants and pass it to StreesModel.makeLabel() function
    # to initialize some stress thresholds

    # if fromNowToGivenTimeToDayNum(campaign_start_time) == (SURVEY_DURATION-1)  and ema_order == 4:
    if (fromNowToGivenTimeToDayNum(campaign_start_time) == SURVEY_DURATION) and (ema_order == 4):
        initStressThresholds(users_info, data_sources) # 0831에 전체 init test할 때는 survey duration 맞춰야함

    # ### Debugging temp
    # initStressThresholds(users_info, data_sources)
    # ### Debugging temp

    user_current_cnt = 0
    test_user_id_list = ["enterqubic@gmail.com", "hscho1226@gmail.com", "hyunsungcho@nmsl.kaist.ac.kr", "ter194@ajou.ac.kr"]
    pilot_user_id_list = ["people9632@gmail.com", "pslzero7@gmail.com", "nrwpath@gmail.com", "kinirotaiyo@gmail.com", "y57h57y@gmail.com",
                         "yoonju095@gmail.com", "kimsoo1357@gmail.com", "lhg0952@gmail.com", "sue712900@gmail.com", "jji1001201@snu.ac.kr",
                         "hyena0228@snu.ac.kr", "asio970425@gmail.com", "harry5414@naver.com", "ha02jj@gmail.com", "joshua124@snu.ac.kr",
                         "zoohonglight@gmail.com", "skrud0212@gmail.com", "thddmcfly2004@gmail.com", "leejongjin71@gmail.com", "dk02315@gmail.com",
                         "sonseho34@gmail.com", "hdlee327@gmail.com", "sudachi98@gmail.com", "jahny1004@gmail.com",
                         "a01080309134@gmail.com", "nanapow2@gmail.com", "kkwtom@naver.com", "shwoals8@gmail.com", "aa020307bjh@gmail.com"]

    users_total_cnt = users_info.__len__() - len(test_user_id_list)

    for user_email, id_jointime in users_info.items():
        # TODO: temporarily check for one user
        if user_email in test_user_id_list:
            continue # test id 제거

        if user_email != "qubic98@gmail.com": # test 용으로 if 놔두기
        # if user_email == 'woghrnt2@ajou.ac.kr':
        # if user_email in pilot_user_id_list: # pilot data 가져오기 용
            user_current_cnt += 1
            user_id = id_jointime['uid']
            day_num = fromNowToGivenTimeToDayNum(id_jointime['joinedTime'])
            print("\nUser {}/{} : {}, Day num : {}".format(user_current_cnt, users_total_cnt, user_email, day_num))
            user_start_time = datetime.datetime.now()
            print("User_start_time: ", user_start_time)
            # check point 1
            threshold_data = grpc_handler.grpc_load_user_data(from_ts=0, uid=user_email,
                                                              data_sources={Features.STRESS_LVL_THRESHOLDS: data_sources[Features.STRESS_LVL_THRESHOLDS]},
                                                              data_src_for_sleep_detection=None)

            stress_lv0_max = 0
            stress_lv1_max = 0
            stress_lv2_min = 0

            if list(threshold_data[Features.STRESS_LVL_THRESHOLDS]).__len__() > 0:
                for item in reversed(threshold_data[Features.STRESS_LVL_THRESHOLDS]):
                    stress_lv0_max, stress_lv1_max, stress_lv2_min = item[1].split(" ")
                    break

            sm = StressModel(user_email, day_num, ema_order, float(stress_lv0_max), float(stress_lv1_max), float(stress_lv2_min))

            # 1. Check if the users day num is 14 and job is for initial model training
            # 데이터 수집 마지막 날, feature extraction & Model init
            if (day_num == SURVEY_DURATION) and (i == FLAG_INITIAL_MODEL_TRAIN):
            # if (day_num >= SURVEY_DURATION) and (i == FLAG_INITIAL_MODEL_TRAIN): # test용
                print("Start 1. Initial model training...", datetime.datetime.now())
                from_time = 0  # from the very beginning of data collection
                try:
                    # grpc data load
                    grpc_start_time = datetime.datetime.now()
                    data = grpc_handler.grpc_load_user_data(from_ts=from_time, uid=user_email,
                                                            data_sources=data_sources,
                                                            data_src_for_sleep_detection=Features.SCREEN_ON_OFF)

                    print("Total grpc_load_user_data... ", datetime.datetime.now() - grpc_start_time)

                    data_size = get_size(data)
                    print("data size: " + str(round(data_size/1024, 2)) + "KB")

                    ### temp
                    grpc_temp_data_PATH = "C:\\Users\\USER\\Desktop\\JH\\code\\Pilot experiment analysis\\data\\grpc_data\\"
                    with open(grpc_temp_data_PATH + user_email + "grpc_data.pkl", "wb") as f:
                        pickle.dump(data, f)
                    ### temp

                    if list(data[Features.SURVEY_EMA]).__len__() >= 3:
                        initialModelTraining(data, user_email, id_jointime['joinedTime'], sm)
                    else:
                        print("Too Less Data!!")
                        continue

                except Exception as e:
                    print("stress_prediction_service.py .....Error in initioalModel", e)

                print("End 1. Initial model training...", datetime.datetime.now())

            # 2. Check if users day num is more than SURVEY_DURATION, only then extract features and make prediction
            # TODO --> day_num >= SURVEY_DURATION  : EMA 수집 마지막날, 서버에 제대로 올라가는지 확인하기 위해
            elif (day_num >= SURVEY_DURATION and i != FLAG_INITIAL_MODEL_TRAIN):
            # 6. Get trained stress prediction model
                try: # Exception during get Feature
                    print("Start 2. Regular prediction... ", datetime.datetime.now())
                    # print("User  {}, Day num : {}".format(user_email, day_num))
                    # 3. Retrieve the last 4 hours data from the gRPC server
                    from_time = now_time - (4 * 3600 * 1000)  # from 4 hours before now time

                    # grpc data load
                    grpc_start_time = datetime.datetime.now()
                    # check point 2, data type: dictionary
                    data = grpc_handler.grpc_load_user_data(from_ts=from_time, uid=user_email,
                                                            data_sources=data_sources,
                                                            data_src_for_sleep_detection=Features.SCREEN_ON_OFF)

                    data_size = get_size(data)
                    print("Total grpc_load_user_data... {0} / data size: {1}KB".format(datetime.datetime.now() - grpc_start_time, str(round(data_size/1024, 2))))

                    # print("data size: " + str(round(data_size/1024, 2)) + "KB")

                    # 4. Extract features from retrieved data
                    extract_feature_start_time = datetime.datetime.now()
                    # print("Start 4. Extract features from retrieved data... ", extract_feature_start_time)

                    # feature extraction이 제대로 안된 경우 대비, 한번 더 체크
                    try:
                        with open('data_result/' + str(user_email) + "_features.p", 'rb') as file:
                            step1_preprocessed = pickle.load(file)

                    except Exception as e:
                        print("Not conduct feature extraction...")
                        initialModelTraining(data, user_email, id_jointime['joinedTime'], sm)
                        with open('data_result/' + str(user_email) + "_features.p", 'rb') as file:
                            step1_preprocessed = pickle.load(file)
                        print("Re feature extraction ok...")

                    features = Features(uid=user_email, dataset=data, joinTimestamp=id_jointime['joinedTime'])

                    df = pd.DataFrame(features.extract_regular(start_ts=from_time, end_ts=now_time,
                                                               ema_order=ema_order, user_email = user_email, day_num=day_num))

                    # 5. Pre-process and normalize the extracted features
                    new_row_preprocessed = sm.preprocessing(df=df, prep_type=None)
                    norm_df = sm.normalizing("new", step1_preprocessed, new_row_preprocessed, user_email, day_num, ema_order)

                    new_row_for_test = norm_df[(norm_df['Day'] == day_num) & (norm_df['EMA order'] == ema_order)]  # get test data

                    ### temp code
                    with open('C:\\Users\\USER\\Desktop\\JH\\shap_check_data\\' + str(user_email) + "_temp_norm_df.p", 'wb') as file:
                        pickle.dump(norm_df, file)
                    print("temp norm_df save...")
                    ### temp code

                    print("Total Extract features from retrieved data... ", datetime.datetime.now() - extract_feature_start_time)

                # Exception during get Feature
                except Exception as e:
                    print("stress_prediction_service.py ..... Exception during get Feature...maybe location related...", e)
                    pass

                # 모델이 저장 안된 경우 대비
                try:
                    with open('model_result/' + str(user_email) + "_model.p", 'rb') as file:
                        initModel = pickle.load(file)

                except Exception as e: # 모델이 저장 안된 경우 대비
                    print("User {} , Model load fail!!!...".format(user_email))
                    print("stress_prediction_service.py .....Excpetion during getInitModel", e)
                    from_time = 0  # from the very beginning of data collection
                    data = grpc_handler.grpc_load_user_data(from_ts=from_time, uid=user_email,
                                                            data_sources=data_sources,
                                                            data_src_for_sleep_detection=Features.SCREEN_ON_OFF)

                    ## Check if EMA response is more than 3
                    if list(data[Features.SURVEY_EMA]).__len__() >= 3:
                        initialModelTraining(data, user_email, id_jointime['joinedTime'], sm)
                        try:
                            with open('model_result/' + str(user_email) + "_model.p", 'rb') as file:
                                initModel = pickle.load(file)
                        except Exception as e:
                            print("Model load error: ", e)
                            pass
                    else:
                        print("Too less Data")
                        continue

                # Exception during saving model result
                try:
                    features = StressModel.feature_df_with_state['features'].values

                    y_pred = initModel.predict(new_row_for_test[features])

                    try: # y_pred 결과가 이상할 때 대비
                        if len(y_pred) > 1:
                            y_pred = y_pred[0]
                    except Exception as e:
                        print("temp if e: ", e)

                    new_row_preprocessed['Stress_label'] = y_pred

                    # 8. Insert a new pre-processed feature entry together with it's predicted label in DB for further model re-train
                    update_df = pd.concat([step1_preprocessed.reset_index(drop=True), new_row_preprocessed.reset_index(drop=True)])
                    update_df = update_df.reset_index(drop=True)

                    # TODO --> Debugging시 주석 처리해야 됨!
                    with open('data_result/' + str(user_email) + "_features.p", 'wb') as file:
                        pickle.dump(update_df, file)

                    print("update_df save...")

                    # 9. Save prediction and important features in DB
                    user_all_labels = list(set(step1_preprocessed['Stress_label']))
                    model_results = list(sm.saveAndGetSHAP(user_all_label=user_all_labels, pred=y_pred,
                                                           new_row_raw = new_row_preprocessed, new_row_norm =new_row_for_test,
                                                           initModel=initModel))  # saves results on ModelResult table in DB

                    # 10. Construct a result message and send it to gRPC server with "STRESS_PREDICTION" data source id
                    construct_start_time = datetime.datetime.now()
                    result_data = {}

                    for model_result in model_results:
                        result_data[str(model_result.prediction_result)] = {
                            "day_num": model_result.day_num,
                            "ema_order": model_result.ema_order,
                            "accuracy": model_result.accuracy,
                            "feature_ids": model_result.feature_ids,
                            "model_tag": model_result.model_tag
                        }

                    try:
                        for i in range(3):
                            if str(i) in result_data.keys():
                                print("result", i, end="\t")

                            else:
                                result_data[str(i)] = {
                                    "day_num": day_num,
                                    "ema_order": ema_order,
                                    "accuracy": 0,
                                    "feature_ids": "NO_FEATURES",
                                    "model_tag": False
                                }

                        print("")
                        # print("stress_pred_service.py ... result: ", result_data)

                    except Exception as e:
                        print("stress_pred_service.py ... Except: ", e)

                    # print("Total construct time... ", datetime.datetime.now() - construct_start_time) # 시간 1초도 안 걸림

                    # TODO : 만약 Test할 때, 서버에 안 올라가게 하려면, grpc_send_user_data 주석처리
                    if result_data:
                        result_grpc_send_start_time = datetime.datetime.now()
                        print("Send to gRPC: result data... ", result_data.values())

                        grpc_handler.grpc_send_user_data(user_id, user_email, data_sources[Features.STRESS_PREDICTION],
                                                        now_time, str(result_data))

                        print("Total result_grpc_send_time... ", datetime.datetime.now() - result_grpc_send_start_time)

                    # 11. Lastly, check if user self reported his stress, then update the DB of pre-processed features with reported stress label
                    self_report_start_time = datetime.datetime.now()
                    check_and_handle_self_report(user_email, data, sm)
                    print("Total self_report_time... ", datetime.datetime.now() - self_report_start_time)

                except Exception as e:
                    print("Exception during saving model result: ", e)
                    pass

                # 7. Make prediction using current extracted features
                print("End", datetime.datetime.now())

            else:
                print("3. Waiting until survey completion day ({} days)...".format(SURVEY_DURATION))

            print("One user total time... ", datetime.datetime.now() - user_start_time)

    grpc_handler.grpc_close()
    print("\nEnd prediction_task time... ", datetime.datetime.now())
    print("Total prediction_task time... ", datetime.datetime.now() - start_prediction_task)

def initialModelTraining(dataset, user_email, joined_timestamp, stress_model):
    initialModelTraining_start_time = datetime.datetime.now()
    # first model init based on 14 days data
    global grpc_handler

    features = Features(uid=user_email, dataset=dataset, joinTimestamp=joined_timestamp)

    df = pd.DataFrame(features.extract_for_after_survey())

    # preprocessing and saving the result
    df_preprocessed = stress_model.preprocessing(df=df, prep_type="default")

    # TODO : Debugging 주석 처리
    with open('data_result/' + str(user_email) + "_features.p", 'wb') as file:
        pickle.dump(df_preprocessed, file)

    # normalizing
    norm_df = stress_model.normalizing("default", df_preprocessed, None, None, None, None)

    # init model
    stress_model.initModel(norm_df)

    print("Total initialModelTraining_start_time... ", datetime.datetime.now() - initialModelTraining_start_time)

def check_and_handle_self_report(user_email, data, stress_model):
    try:
        # 'SELF_STRESS_REPORT' data source in gRPC server holds self reported stress from users
        sr_day_num = 0
        sr_ema_order = 0
        sr_value = -1  # self report value

        if data['SELF_STRESS_REPORT']:  # data['SELF_STRESS_REPORT'][-1][1] takes the value of the latest SELF_STRESS_REPORT data source
            if data['SELF_STRESS_REPORT'][-1][1]:
                timestamp, sr_day_num, sr_ema_order, yes_no, sr_value = [int(i) for i in data['SELF_STRESS_REPORT'][-1][1].split(" ")]
                # sr_vale == 유저가 응답한 스트레스 레벨
                # timestamp, day_num, order, yesOrNo, reportAnswer
                model_result_to_update = ModelResult.objects.get(uid=user_email, day_num=sr_day_num, ema_order=sr_ema_order, prediction_result=sr_value)
                update_day_num = model_result_to_update.day_num
                update_ema_order = model_result_to_update.ema_order
                update_user_tag = model_result_to_update.user_tag

                # 0902 수정
                stress_model.update(sr_value, update_day_num, update_ema_order, update_user_tag)

                # 기존 코드
                # # check if this result was not already updated by the user, if it wasn't then update the user tag and re-train the model
                # if model_result_to_update.user_tag == False:
                #     stress_model.update(sr_value, model_result_to_update.day_num, model_result_to_update.ema_order)

    except Exception as e:
        print("check_and_handle_self_report error: ", e)

def fromNowToGivenTimeToDayNum(givenTimestamp):
    joindate = datetime.datetime.fromtimestamp(givenTimestamp / 1000)
    joindateAtStart = joindate.replace(hour=0, minute=0)
    nowTime = int(datetime.datetime.now().timestamp())
    nowDate = datetime.datetime.fromtimestamp(nowTime)

    return int((nowDate - joindateAtStart).days)

def initStressThresholds(users_info, data_sources):
    initStressThresholds_start_time = datetime.datetime.now()
    global grpc_handler
    # all_stress_level_values = []
    for user_email, id_jointime in users_info.items():
        user_stress_level_value_list = [] # user 별로 stress value 저장
        data = grpc_handler.grpc_load_user_data(from_ts=0, uid=user_email,
                                                data_sources={Features.SURVEY_EMA: data_sources[Features.SURVEY_EMA]},
                                                data_src_for_sleep_detection=None)
        ema_responses = list(data[Features.SURVEY_EMA])

        # ema_responses에 중복된 값이 있을 수 있으므로 사전 형식으로 중복 제거
        ema_responses_dict = dict(ema_responses)
        ema_responses = list(ema_responses_dict.items())
        print("ema_responses: ", len(ema_responses))

        if ema_responses.__len__() > 0:
            for index, ema_res in enumerate(ema_responses):
                values = ema_res[1].split(" ")
                responded_time = values[0]
                ema_order = values[1]
                ans1 = values[2]
                ans2 = values[3]
                ans3 = values[4]
                ans4 = values[5]
                user_stress_level_value_list.append(int(ans1) + int(ans2) + int(ans3) + int(ans4))

        threshold_results = makeLabel(user_stress_level_value_list)
        final_stress_threshold_results = "{} {} {}".format(threshold_results[0], threshold_results[1], threshold_results[2])
        # print("final_stress_threshold_results: ", final_stress_threshold_results)
        now_time = int(datetime.datetime.now().timestamp() * 1000)

        grpc_handler.grpc_send_user_data(id_jointime['uid'], user_email,
                                         data_sources[Features.STRESS_LVL_THRESHOLDS],
                                         now_time, final_stress_threshold_results)

    print("Total initStressThresholds time...", datetime.datetime.now() - initStressThresholds_start_time)
