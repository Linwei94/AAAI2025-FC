import torch


def train_default(train_loader, model, device, optimizer, criterion, epoch, args):
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if len(target.shape) == 2:
            target = target.argmax(dim=1)
        optimizer.zero_grad()
        output = model(data)
        if args.criterion == 'correlation_loss': 
            loss = criterion(output, target, type)
        else:
            loss = criterion(output, target)
        loss.backward()
        if args.clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()
        if i % 50 == 0:
            print(f"Epoch: {epoch + 1}/{args.train_epochs}  "
                f"Step: {i}  "
                f"Loss: {loss.item()}")
            


def train_hybrid(train_loader, model, device, optimizer, criterion, epoch, args):
    for i, (data_ori, target_ori, data_mix, target_mix) in enumerate(train_loader):
        data_ori, target_ori = data_ori.to(device), target_ori.to(device)
        data_mix, target_mix = data_mix.to(device), target_mix.to(device)
        optimizer.zero_grad()
        data = torch.cat((data_ori, data_mix)).to(device)
        output = model(data)
        output_ori, output_mix = torch.split(output, data_ori.shape[0])
        loss = criterion(output_ori, output_mix, target_ori, target_mix)
        loss.backward()
        if args.clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()
        if i % 50 == 0:
            print(f"Epoch: {epoch + 1}/{args.train_epochs}  "
                f"Step: {i}  "
                f"Loss: {loss.item()}")