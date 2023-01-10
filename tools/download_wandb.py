import wandb

api = wandb.Api()

runs = api.runs("weakly_guys/weakvg", {'created_at': {"$gt": '2023-01-08T##'}})

for run in runs:
  history = run.history(pandas=False)

  max_acc_val = 0
  max_epoch = 0
  max_point_val = 0

  for step in history:
    acc_val = round(step['acc_val'] * 100, 1)
    point_val = round(step['point_acc_val'] * 100, 1)
    epoch = step['_step'] + 1

    if acc_val > max_acc_val:
      max_acc_val = acc_val
      max_epoch = epoch
      max_point_val = point_val
  
  print(f"{run.notes} -> {max_acc_val} ({max_epoch}), {max_point_val}")

