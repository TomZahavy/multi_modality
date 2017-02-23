import csv
# foldername = 'image_res/'
foldername = 'max/'
# foldername = 'policy_top3/'
# filenames = ['L_val','L_train','LF_train','LF_val','L2_train','L2_val','F_train','F_val']foldername = 'image_res/'
# filenames = ['title_val','title_train','image_train','image_val','oracle_train','oracle_val','policy_train','policy_val','policy_pred_train','policy_pred_val']
# filenames= ['3_logits_train','3_logits_val','logits_train','logits_val']
# filenames= ['top1_train','top1_val','top3_train','top3_val']
# filenames= ['top1_val','top1_title_val','top3_val','top3_title_val']
# filenames = ['title_validation','title_train','image_train','image_validation','opt_train','opt_validation','policy_train','policy_validation','policy_pred_train','policy_pred_validation']
filenames = ['title_validation','image_validation','opt_validation','policy_validation','policy_pred_validation']
for filename in filenames:
    with open('/Users/tzahavy/Desktop/'+foldername+filename+'.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        count = 0
        acc_vals = []
        for row in spamreader:
            if count>0:
                vals = row[0].split(",")
                acc_vals.append(float(vals[-1]))
            count+=1

    n = len(acc_vals)

    last_vals = acc_vals[n-4:n-2]
    score = sum(last_vals) / float(len(last_vals))
    print filename+ ' score: ' +str(score)