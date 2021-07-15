
from rlpyt.utils.collections import namedarraytuple

AgentInfo = namedarraytuple("AgentInfo", ["dist_info", "value"])
AgentInfoTwin = namedarraytuple("AgentInfoTwin", ["dist_info", "dist_int_info", "value", "int_value"])
AgentInfoRnn = namedarraytuple("AgentInfoRnn", ["dist_info", "value", "prev_rnn_state"])
AgentInfoRnnTwin = namedarraytuple("AgentInfoRnnTwin", [
                                    "dist_info", "dist_int_info", 
                                    "value", "int_value", 
                                    "prev_rnn_state", "prev_int_rnn_state"])
IcmInfo = namedarraytuple("IcmInfo", [])
NdigoInfo = namedarraytuple("NdigoInfo", ["prev_gru_state"])
RndInfo = namedarraytuple("RndInfo", [])