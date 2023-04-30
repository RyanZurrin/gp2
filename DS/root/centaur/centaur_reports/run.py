import json
from centaur_reports.report_sr import StructuredReport


# def dbt_files(dcm_dir, study):
#     ds_list = {'DXm': {}, 'BT': {}}
#     for ds_dir in os.listdir(os.path.join(dcm_dir, study)):
#         if 'DXm' in ds_dir:
#             this_ds = utils.dh_dcmread('{}/{}'.format(os.path.join(dcm_dir, study), ds_dir))
#             ds_list['DXm'][this_ds.SOPInstanceUID] = this_ds
#         elif 'BT' in ds_dir:
#             this_ds = utils.dh_dcmread('{}/{}'.format(os.path.join(dcm_dir, study), ds_dir))
#             ds_list['BT'][this_ds.SOPInstanceUID] = this_ds
#         else:
#             continue
#     preds = json.load(open(dcm_dir + '/' + study + '.json'))
#
#     overall_risk = preds['overall_score']
#     del preds['overall_score']
#
#     for k in list(preds.keys()):
#         preds[k]['score'] = [float(preds[k]['score'])]
#
#     preds_list = {'DXm': {}, 'BT': {}}
#
#     for pred in preds:
#         if 'DXm' in pred:
#             preds_list['DXm']['.'.join(pred.split('.')[1:])] = preds[pred]
#         elif 'BT' in pred:
#             preds_list['BT']['.'.join(pred.split('.')[1:])] = preds[pred]
#
#     return ds_list, preds_list, overall_risk
#
#
# host_dir = '/Users/kevinwu/deephealth/data/dh_dh0new'
# studies = [
#     '1.2.826.0.1.3680043.9.7134.1.2.0.23161.1505941466.143485']
# # '1.2.826.0.1.3680043.9.7134.1.2.0.59097.1505864312.795666']
# # '1.2.826.0.1.3680043.9.7134.1.2.0.72830.1505877232.495996']
#
# for study in studies:
#     ds_list_all, preds_list_all, overall_risk = dbt_files(
#         host_dir, study)
#
#     for image in ['DXm', 'BT']:
#         ds_list = ds_list_all[image]
#         preds_list = preds_list_all[image]
#         srwriter = SRWriter(ds_list, [preds_list, overall_risk])
#         srwriter.populate_sr()
#         save_path = host_dir + '/' + 'DH_CAD_{}_'.format(image) + study
#         srwriter.save(save_path)
#         print('Saved to {}'.format(save_path))
#
# pdb.set_trace()

if __name__ == '__main__':
    file_path = '/efs/ericwu/docker/centaur_v5/outputs/test4/' \
                '1.2.826.0.1.3680043.9.7134.1.2.0.10050.1505921978.734930/results.json'
    results_json = json.load(open(file_path, 'rb'))
    output_dir = '/home/kevinwu/centaur_reports/'

    srwriter = StructuredReport()

    srwriter.generate(results_json['model_results'], results_json['metadata'], output_dir)