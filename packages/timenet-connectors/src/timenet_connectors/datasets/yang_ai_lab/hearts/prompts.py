"""HEARTS prompts adapted from ``exp/<source>/<task>.py`` in the pinned release.

File references name TimeF signals. Input annotations contain case-specific values.
"""

A1C_PROMPT = (
    "The continuous glucose monitors (CGM) data for this subject is in signal "
    "'window_df.Libre GL', in mg/dL. Your task is to predict the disease status based on CGM "
    "data. There are 3 different status: normal, prediabetes, and diabetes. Please analyze the "
    "entire CGM time series and output your final prediction of disease status as a JSON object "
    'without any other text in the following format:\n{\n    "disease_status": [string, '
    "normal|prediabetes|diabetes]\n}"
)
CGM_STAT_PROMPT = (
    "The continuous glucose monitors (CGM) data for this subject is in signal 'cgm.Libre GL', "
    "in mg/dL. Calculate percentage of time CGM is below and above normal range (70 - 180 "
    "mg/dL). Please calculate and output your final answer as a JSON object without any other "
    'text in the following format:\n{\n    "below": [float, percentage of time CGM < 70 mg/dL],'
    '\n    "above": [float, percentage of time CGM > 180 mg/dL]\n}'
)
FASTING_GLU_PROMPT = (
    "The continuous glucose monitors (CGM) data for this subject is in signal "
    "'window_df.Libre GL', in mg/dL. Your task is to predict the subject's fasting blood glucose "
    "value (in mg/dL) based on the entire CGM time series. Please output your final prediction as "
    'a JSON object without any other text in the following format:\n{\n    "fasting_glu": [float, '
    "fasting blood glucose value in mg/dL]\n}"
)
IAUC_PROMPT = (
    "The CGM (continuous glucose monitoring) data for 2 hours after a meal is in signal "
    "'cgm_df.CGM (mg/dL)', in mg/dL, and the signal's time axis gives the minutes since the meal "
    "started.\nPlease calculate the incremental Area Under the Curve (iAUC) for the postprandial "
    "glucose response. Use first CGM value as baseline. Please output your calculated iAUC value "
    'as a JSON object without any other text in the following format:\n{\n    "iauc": [number]\n}'
)
MEAL_REACT_PROMPT = (
    "You are given two CGM data windows (A and B), each is a 4-hour window (1 hour before and 3 "
    "hours after a meal) from two different subjects. Both subjects ate a meal with similar "
    "carbohydrate and calorie content, but one subject is normal (non-diabetes) and the other has "
    "prediabetes or diabetes. The windows are in signals 'A.Libre GL' and 'B.Libre GL', in "
    "mg/dL.\n\nYour task: Based on the CGM data, decide which window (A or B) is from the normal "
    "subject and which is from the prediabetes or diabetes subject. Output your answer as a JSON "
    'object with the following format (no extra text):\n{\n    "normal_subject": "A"  # or "B"\n}'
)
MEAL_TIME_PROMPT = (
    "The continuous glucose monitors (CGM) data for this subject is in signal "
    "'window_df.Libre GL', in mg/dL, and the signal's time axis gives the time of each reading "
    "(in integer minutes). There is exactly one meal event in this 2-hour window. Please analyze "
    "the CGM data and output your final answer as a JSON object without any other text in the "
    "following format:\n"
    '{\n    "meal_timestamp": [float, timestamp (in minutes) when the meal starts]\n}'
)
COSWARA_AUDIO_CLASS_PROMPT = (
    "Analyze the audio in signal 'data.signal'.\n\nClassify the audio as one of the following "
    "categories:\n- speech: spoken words or sounds like counting or vowels\n- cough: coughing "
    "sounds\n- breathing: breathing sounds (deep or shallow)\n\nOutput your classification in "
    'JSON format:\n{\n    "prediction": "speech" or "cough" or "breathing"\n}'
)
COSWARA_COUGH_PROMPT = (
    "Analyze the cough audio in signal 'data.signal'.\n\nBased on the sound of the cough, "
    "classify the subject as either healthy or covid positive.\n\nOutput your classification in "
    'JSON format:\n{\n    "prediction": "healthy" or "covid_positive"\n}'
)
COSWARA_COUGH_SYMPTOMS_PROMPT = (
    "Analyze the cough audio in signal 'data.signal' and the subject's symptoms to classify the "
    "subject as either healthy or covid positive.\n\nThe symptoms are in the 'symptoms' "
    "annotation.\n\nBased on the sound of the cough and the symptoms, classify the subject as "
    "either healthy or covid positive.\n\nOutput your classification in JSON format:\n{\n    "
    '"prediction": "healthy" or "covid_positive"\n}'
)
COSWARA_SPEECH_PROMPT = (
    "Analyze the speech audio in signal 'data.signal'.\n\nBased on the sound of the speech, "
    "classify the subject as either healthy or covid positive.\n\nOutput your classification in "
    'JSON format:\n{\n    "prediction": "healthy" or "covid_positive"\n}'
)
COUGHVID_DETECTION_PROMPT = (
    "One audio recording is in signal 'audio'. It is a cough audio sampled at 48 kHz. Analyze "
    "the audio and determine if a cough is present in the recording.\n\nOutput your answer in the "
    'following JSON format without any other text:\n{\n    "has_cough": true/false\n}'
)
COUGHVID_COVID_PROMPT = (
    "One audio recording is in signal 'audio', which is sampled at 48 kHz. Analyze the audio and "
    "determine if the subject has COVID-19.\n\nOutput your answer in the following JSON format "
    'without any other text:\n{\n    "is_covid": true/false\n}'
)
COUGHVID_DIAGNOSIS_PROMPT = (
    "One audio recording is in signal 'audio', which is sampled at 48 kHz. Analyze the audio and "
    "determine the most likely diagnosis from the following options:\n- upper_infection\n- "
    "lower_infection\n- obstructive_disease\n- COVID-19\n- healthy_cough\n\nOutput your answer in "
    'the following JSON format without any other text:\n{\n    "diagnosis": "diagnosis_option"\n}'
)
COUGHVID_HEALTH_PROMPT = (
    "One audio recording is in signal 'audio', which is sampled at 48 kHz. Analyze the audio and "
    "determine if the subject is healthy.\n\nHealthy is defined as the subject having no "
    "underlying health conditions can be recognized from the audio.\n\nOutput your answer in the "
    'following JSON format without any other text:\n{\n    "is_healthy": true/false\n}'
)
COUGHVID_MFCC_PROMPT = (
    "One audio recording is in signal 'audio'. It is a cough audio sampled at 48 kHz. Calculate "
    "the Mel Frequency Cepstral Coefficients (MFCCs) with 13 coefficients, then compute the mean "
    "and standard deviation of each MFCC over time. Please output your final answer in the "
    'following JSON format without any other text:\n{\n    "mfcc_mean": [list of 13 floats],\n    '
    '"mfcc_std": [list of 13 floats]\n}'
)
ALTITUDE_RESPIRATION_PROMPT = (
    "You are given 5-minute segments of respiration signals in signals 'segment_dfs.A.rsp', "
    "'segment_dfs.B.rsp' and 'segment_dfs.C.rsp'. Each segment corresponds to one of the altitude "
    "ranges provided below:\n\n- 1.5k-2k meters\n- 2k-2.5k meters\n- 2.5k-3k meters\n- 3k-3.5k "
    "meters\n- 3.5k-4k meters\n\nYour task is to analyze the respiration signals and rank the "
    "segments from highest altitude to lowest altitude.\n\nOutput your ranking as a list of "
    'labels in JSON format:\n{\n    "ranking": ["label from highest altitude", "label from mid '
    'altitude", "label from lowest altitude"]\n}'
)
ALTITUDE_SPO2_PROMPT = (
    "You are given 5-minute segments of SpO2 (oxygen saturation) signals in signals "
    "'segment_dfs.A.spo', 'segment_dfs.B.spo' and 'segment_dfs.C.spo'. Each segment corresponds "
    "to one of the altitude ranges provided below:\n\n- 1.5k-2k meters\n- 2k-2.5k meters\n- "
    "2.5k-3k meters\n- 3k-3.5k meters\n- 3.5k-4k meters\n\nYour task is to analyze the SpO2 "
    "signals and rank the segments from highest altitude to lowest altitude.\n\nOutput your "
    'ranking as a list of labels in JSON format:\n{\n    "ranking": ["label from highest '
    'altitude", "label from mid altitude", "label from lowest altitude"]\n}'
)
HR_RESP_PAIRING_PROMPT = (
    "You are given 5-minute segments of respiration and heart rate signals from the same subject "
    "at different altitude phases.\n\nThe respiration signals are in signals "
    "'respiration_dfs.respiration_A.rsp' and 'respiration_dfs.respiration_B.rsp'.\n\nThe heart "
    "rate signals are in signals 'hr_dfs.hr_1.hr' and 'hr_dfs.hr_2.hr'.\n\nYour task is to "
    "analyze the signals and pair which heart rate signal corresponds to which respiration "
    "signal, based on them being from the same altitude phase. The pairing should reflect which "
    "heart rate matches which respiration from the same phase.\n\nOutput your pairing as a "
    "dictionary in JSON format without any other text, for example:\n{\n    "
    '"pairing": {"A": "1", "B": "2"} # or {"A": "2", "B": "1"}\n}'
)
SPO2_RESP_PAIRING_PROMPT = (
    "You are given 5-minute segments of respiration and SpO2 (oxygen saturation) signals from the "
    "same subject at different altitude phases.\n\nThe respiration signals are in signals "
    "'respiration_dfs.respiration_A.rsp' and 'respiration_dfs.respiration_B.rsp'.\n\nThe SpO2 "
    "signals are in signals 'spo_dfs.spo_1.spo' and 'spo_dfs.spo_2.spo'.\n\nYour task is to "
    "analyze the signals and pair which SpO2 signal corresponds to which respiration signal, "
    "based on them being from the same altitude phase. The pairing should reflect which SpO2 "
    "matches which respiration from the same phase.\n\nOutput your pairing as a dictionary in "
    'JSON format without any other text, for example:\n{\n    "pairing": {"A": "1", "B": "2"} # '
    'or {"A": "2", "B": "1"}\n}'
)
VCTK_DIRECTION_PROMPT = (
    "A raw audio waveform is in signal 'waveform'. This is a mono audio signal sampled at 16000 "
    "Hz. The waveform may be playing forward (normal) or time-reversed (backward). Determine "
    "whether the waveform is playing in the forward direction or has been time-reversed. Output "
    "your final answer as a JSON object without any other text, in the following format:\n{\n    "
    '"direction": "[forward|reversed]",\n    "reason": "[explanation of your choice]"\n}'
)

FORECAST_PROMPT = (
    "Forecast the next 30 one-minute CGM values after the meal time. Use the supplied CGM window "
    "and any reference data or meal details in the input annotations. Return the values in time order."
)
IMPUTE_PROMPT = (
    "Impute the 30 corrupted CGM readings between mask_start and mask_end. The input window keeps "
    "the source's zeroed segment and any extra activity or heart-rate column. Return values in time order."
)
MEAL_IMAGE_PROMPT = (
    "A meal occurred at meal_time in the input annotations. Use the four-hour CGM window and the "
    "images a.jpg, b.jpg, c.jpg, and d.jpg to choose the most likely meal image."
)
SYMPTOMS_ONLY_PROMPT = "Classify COVID status as healthy or covid_positive using only the supplied symptoms."
