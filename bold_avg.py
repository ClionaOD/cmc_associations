import pickle
import os
import numpy as np

path = '/data/movie-associations/semantic_measures/imgnet_bold_rdms_per-subj_per-roi.pickle'

with open(path,'rb') as f:
    bold_rdms = pickle.load(f)

bold_rdms = {k:v for k,v in bold_rdms.items() if not k == 'CSI4'}

rois=['LHEarlyVis', 'LHLOC', 'LHOPA', 'LHPPA', 'LHRSC', 'RHEarlyVis', 'RHLOC', 'RHOPA', 'RHPPA', 'RHRSC']
avg_rdms = {roi:[] for roi in rois}

for subj, roi_dict in bold_rdms.items():
    for roi, df in roi_dict.items():
        avg_rdms[roi].append(df)

for roi,lst in avg_rdms.items():
    avg_rdms[roi] = np.mean(np.stack(avg_rdms[roi]),axis=0)

with open('./across_subj_bold_imgnet_rdms.pickle','wb') as f:
    pickle.dump(avg_rdms,f)

save_path = "/data/movie-associations/brain_rdms/BOLD5000/imgnet_1916"
for roi, arr in avg_rdms.items():
    np.savetxt(f"{save_path}/{roi}_rdm_across_subj.csv", arr, delimiter=",")