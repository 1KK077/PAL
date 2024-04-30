import torch
import numpy as np
from PIL import Image
from apex import amp
from ignite.engine import Engine
from ignite.engine import Events
from torch.autograd import no_grad
import os
import torchvision.transforms.functional as Ff
from utils.calc_acc import calc_acc
from torch.nn import functional as F

# def some_function(epoch, initial_weight_decay):
#     if epoch > 20:
#         new_weight_decay = initial_weight_decay/100
#     elif epoch > 10 and epoch <= 20:
#         new_weight_decay = initial_weight_decay*1/10
#     elif epoch > 5 and epoch <= 10:
#         new_weight_decay = initial_weight_decay*1/10
#     else:
#         new_weight_decay = initial_weight_decay
#     return new_weight_decay

def some_function(epoch, initial_weight_decay):
    if epoch > 15:
        new_weight_decay = initial_weight_decay/100
    elif epoch > 5 and epoch <= 15:
        new_weight_decay = initial_weight_decay*1/10
    else:
        new_weight_decay = initial_weight_decay
    return new_weight_decay

def create_train_engine(model, optimizer, non_blocking=False):
    device = torch.device("cuda", torch.cuda.current_device())

    def _process_func(engine, batch):
        model.train()

        data, labels, cam_ids, img_paths, img_ids = batch
        epoch = engine.state.epoch

        data = data.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        cam_ids = cam_ids.to(device, non_blocking=non_blocking)
        iteration = engine.state.iteration

        import pdb
        pdb.set_trace()



        if epoch < 21:
            # 进行warmup，逐渐增加学习率
            # lr = 0.00035 * epoch / 20
            lr = 0.00035 * iteration / (30 * 213)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if True == False:  # warmup
            if epoch < 21:
                # 进行warmup，逐渐增加学习率
                # lr = 0.00035 * epoch / 20
                lr = 0.00035 * iteration / (30 * 213)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr



            # for param_group in optimizer.param_groups:
            #     for name, value in param_group.items():
            #         if name != 'params':  # 'params' key contains the model parameters, which we want to exclude
            #             print(name, value)

            if True:
                new_weight_decay = some_function(epoch, 0.5)  # Define your own function
                for param_group in optimizer.param_groups:
                    param_group['weight_decay'] = new_weight_decay

        # for name, param in model.named_parameters():
        #     if 'Prompter' in name:
        #         print(f"Parameter: {name}, Gradient: {param.grad}")
        # import pdb
        # pdb.set_trace()

        optimizer.zero_grad()

        loss, metric = model(data, labels,
                             cam_ids=cam_ids,
                             epoch=epoch)


        # import pdb
        # pdb.set_trace()
        # if epoch == 4:
        #     import pdb
        #     pdb.set_trace()
        #     print(acc_diff)

        # if epoch > 5 and iteration % 5 == 0:
        #     for param_group in optimizer.param_groups:
        #         if acc_diff > 0: #or acc_diff = 0
        #             new_weight_decay = param_group['weight_decay'] * 1.005
        #         else:
        #             new_weight_decay = param_group['weight_decay'] / 2
        #     param_group['weight_decay'] = new_weight_decay

        # l1_lambda = 0.005
        # l1_norm = sum(p.abs().sum() for p in model.parameters())
        # loss += l1_lambda * l1_norm

        loss = torch.mean(loss)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # for name, param in model.named_parameters():
        #     print(name)
        # import pdb
        # pdb.set_trace()

        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


        # grad_classifier = model.classifier.weight.grad
        # print(grad_classifier)
        # import pdb
        # pdb.set_trace()

        optimizer.step()


        return metric

    return Engine(_process_func)


def create_eval_engine(model, non_blocking=False):
    device = torch.device("cuda", torch.cuda.current_device())

    def _process_func(engine, batch):
        model.eval()

        data, labels, cam_ids, img_paths = batch[:4]

        data = data.to(device, non_blocking=non_blocking)

        cam_ids = cam_ids.to(device, non_blocking=non_blocking)
        sub = (cam_ids == 3) + (cam_ids == 6)  # 0 visible

        import pdb
        pdb.set_trace()
        if torch.any(sub == 0):
            import pdb
            pdb.set_trace()




        # image_tensor = data[2]  # 从数据批处理中选择指定索引的图像张量
        #
        # # 将图像张量转换为 PIL 图像
        # image_pil = Ff.to_pil_image(image_tensor)
        #
        # # 指定保存路径
        # save_dir = "/home/zhang/E/RKJ/MAPnet/new-classifier-prompt/prompt"
        # os.makedirs(save_dir, exist_ok=True)  # 确保保存文件夹存在，如果不存在则创建
        #
        # # 保存图像
        # save_path = os.path.join(save_dir, "saved_image1.jpg")
        # image_pil.save(save_path)  # 你可以更改文件名和格式
        #
        # import pdb
        # pdb.set_trace()

        with no_grad():
            feat = model(data, cam_ids=cam_ids.to(device, non_blocking=non_blocking))


        return feat.data.float().cpu(), labels, cam_ids, np.array(img_paths)

    engine = Engine(_process_func)

    @engine.on(Events.EPOCH_STARTED)
    def clear_data(engine):
        # feat list
        if not hasattr(engine.state, "feat_list"):
            setattr(engine.state, "feat_list", [])
        else:
            engine.state.feat_list.clear()

        # id_list
        if not hasattr(engine.state, "id_list"):
            setattr(engine.state, "id_list", [])
        else:
            engine.state.id_list.clear()

        # cam list
        if not hasattr(engine.state, "cam_list"):
            setattr(engine.state, "cam_list", [])
        else:
            engine.state.cam_list.clear()

        # img path list
        if not hasattr(engine.state, "img_path_list"):
            setattr(engine.state, "img_path_list", [])
        else:
            engine.state.img_path_list.clear()

    @engine.on(Events.ITERATION_COMPLETED)
    def store_data(engine):
        engine.state.feat_list.append(engine.state.output[0])
        engine.state.id_list.append(engine.state.output[1])
        engine.state.cam_list.append(engine.state.output[2])
        engine.state.img_path_list.append(engine.state.output[3])

    return engine
