import models.TGL as TGLModel
from utils.config_utils import *
from utils.data_utils import COVIDDataset, COVIDGraphDataset, GraphBatchData, data_loader
from utils.evaluation_utils import evaluate_model_us_covid_dataset

configs = load_configs('./config_usa-covid.json')
init_seed(configs)

dataset_root_path = './datasets/us-covid/'
original_dataset_file_path = './datasets/us-covid/vn62p.csv'
dataset_file_path = './datasets/us-covid/R_fifty_states.csv'
relationship_file_path = './datasets/us-covid/us-covid_neighborhood_graph.csv'

testing_start_at = 298
validation_start_at = 200

dataset = COVIDDataset(root=dataset_root_path,
                       dataset_file_path=dataset_file_path,
                       relationship_file_path=relationship_file_path,
                       configs=configs)
dataset.process()
graph_dataset = COVIDGraphDataset(dataset_file_path=dataset_file_path,
                                  relationship_file_path=relationship_file_path)

graph_dataset_batch = GraphBatchData.from_data_list(graph_dataset)

total_dataset = dataset[:testing_start_at]
test_dataset = dataset[testing_start_at:]
train_dataset = total_dataset[:validation_start_at]
val_dataset = total_dataset[validation_start_at:]

model_TGL = TGLModel.TGL(configs, graph_dataset_batch)

print('Evaluating COVID-19 prediction task with TGL model...')

print("Total number of parameters of model TGL:", sum(p.numel() for p in model_TGL.parameters()))

optimizer = torch.optim.Adam(model_TGL.parameters(), lr=configs.learning_rate)

train_loader = data_loader(dataset=train_dataset, batch_size=configs.batch_size)
val_loader = data_loader(dataset=val_dataset, batch_size=configs.batch_size)
test_loader = data_loader(dataset=test_dataset, batch_size=configs.batch_size)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                       patience=configs.patience,
                                                       threshold=configs.threshold)

print('Starting to train model on the train [{}] / validation [{}] sets...'.format(len(train_dataset),
                                                                                   len(val_dataset)))
losses_tr = []
losses_vl = []
for epoch in range(configs.number_of_epochs):
    loss_tr, h_tr = TGLModel.training(data_loader=train_loader, model=model_TGL)
    losses_tr.append(loss_tr)
    loss_vl, h_vl = TGLModel.validation(data_loader=val_loader, model=model_TGL)
    losses_vl.append(loss_vl)

    scheduler.step(loss_tr)

    if epoch % 10 == 0:
        print(f"Epoch: [{epoch}] | Train Loss: [{loss_tr}]")

print('Done !')
print('Training is completed, starting to evaluate model in testing set [{}]...'.format(len(test_dataset)))
final_mape_dict = dict()
for batch_idx, batch in enumerate(test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        batch.to(device)
        pred, embed = model_TGL(batch.x.float(), batch.edge_index, batch.batch)
        mape_score = evaluate_model_us_covid_dataset(batch, pred)
        final_mape_dict[batch_idx] = mape_score

average_mape = np.mean(list(final_mape_dict.values()))
std_mape = np.std(list(final_mape_dict.values()))
print(f"Final results: (Mean+SD MAPE): {average_mape}")
