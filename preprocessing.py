import pandas as pd
from sklearn.preprocessing import LabelEncoder

columns = (['duration'
,'protocol_type'
,'service'
,'flag'
,'src_bytes'
,'dst_bytes'
,'land'
,'wrong_fragment'
,'urgent'
,'hot'
,'num_failed_logins'
,'logged_in'
,'num_compromised'
,'root_shell'
,'su_attempted'
,'num_root'
,'num_file_creations'
,'num_shells'
,'num_access_files'
,'num_outbound_cmds'
,'is_host_login'
,'is_guest_login'
,'count'
,'srv_count'
,'serror_rate'
,'srv_serror_rate'
,'rerror_rate'
,'srv_rerror_rate'
,'same_srv_rate'
,'diff_srv_rate'
,'srv_diff_host_rate'
,'dst_host_count'
,'dst_host_srv_count'
,'dst_host_same_srv_rate'
,'dst_host_diff_srv_rate'
,'dst_host_same_src_port_rate'
,'dst_host_srv_diff_host_rate'
,'dst_host_serror_rate'
,'dst_host_srv_serror_rate'
,'dst_host_rerror_rate'
,'dst_host_srv_rerror_rate'
,'attack'
,'level'])


df_train=pd.read_csv('../../nsl-kdd/KDDTrain+.txt',header=None,names=columns)
df_test=pd.read_csv('../../nsl-kdd/KDDTest+.txt',header=None,names=columns)

# No duplicate data in the dataset
# No null data

# Classify attack types to attack and non-attack
df_train['bi_attack'] = df_train.attack.map(lambda val: 'normal' if val == 'normal' else 'abnormal')
df_train.drop('attack', axis=1, inplace=True)
df_test['bi_attack'] = df_test.attack.map(lambda val: 'normal' if val == 'normal' else 'abnormal')
df_test.drop('attack', axis=1, inplace=True)

# Encode objects
le = LabelEncoder()
cols = ['protocol_type', 'service', 'flag', 'bi_attack']
for col in cols:
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.fit_transform(df_test[col])

