import torch
from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_accuracy

def train_epoch_multimodal(epoch, data_loader, model, criterion, optimizer, opt,
                           epoch_logger, batch_logger, train_losses, train_prec1s):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end_time = time.time()
    for i, (audio_inputs, visual_inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        # Spostiamo i tensori sul dispositivo corretto
        audio_inputs = audio_inputs.to(opt.device)
        visual_inputs = visual_inputs.to(opt.device)
        targets = targets.to(opt.device)

        # Controlli per maschere (come nel codice precedente)
        if opt.mask is not None:
            with torch.no_grad():
                if opt.mask == 'noise':
                    audio_inputs = torch.cat(
                        (audio_inputs, torch.randn(audio_inputs.size()).to(opt.device), audio_inputs), dim=0)
                    visual_inputs = torch.cat(
                        (visual_inputs, visual_inputs, torch.randn(visual_inputs.size()).to(opt.device)), dim=0)
                    targets = torch.cat((targets, targets, targets), dim=0)
                    shuffle = torch.randperm(audio_inputs.size()[0])
                    audio_inputs = audio_inputs[shuffle]
                    visual_inputs = visual_inputs[shuffle]
                    targets = targets[shuffle]

                elif opt.mask == 'softhard':
                    coefficients = torch.randint(low=0, high=100, size=(audio_inputs.size(0), 1, 1)) / 100
                    vision_coefficients = 1 - coefficients
                    coefficients = coefficients.repeat(1, audio_inputs.size(1), audio_inputs.size(2))
                    vision_coefficients = vision_coefficients.unsqueeze(-1).unsqueeze(-1).repeat(1,
                                                                                                 visual_inputs.size(1),
                                                                                                 visual_inputs.size(2),
                                                                                                 visual_inputs.size(3),
                                                                                                 visual_inputs.size(4))

                    audio_inputs = torch.cat((audio_inputs, audio_inputs * coefficients.to(opt.device),
                                              torch.zeros(audio_inputs.size()).to(opt.device), audio_inputs), dim=0)
                    visual_inputs = torch.cat((visual_inputs, visual_inputs * vision_coefficients.to(opt.device),
                                               visual_inputs, torch.zeros(visual_inputs.size()).to(opt.device)), dim=0)

                    targets = torch.cat((targets, targets, targets, targets), dim=0)
                    shuffle = torch.randperm(audio_inputs.size()[0])
                    audio_inputs = audio_inputs[shuffle]
                    visual_inputs = visual_inputs[shuffle]
                    targets = targets[shuffle]

        # Permute and reshape visual inputs
        visual_inputs = visual_inputs.permute(0, 2, 1, 3, 4)
        visual_inputs = visual_inputs.reshape(visual_inputs.shape[0] * visual_inputs.shape[1], visual_inputs.shape[2],
                                              visual_inputs.shape[3], visual_inputs.shape[4])

        try:
            # Model forward pass
            outputs = model(audio_inputs, visual_inputs)

            # Compute loss
            loss = criterion(outputs, targets)
            print(f"Loss: {loss.item()}")

            losses.update(loss.data, audio_inputs.size(0))

            # Calculate accuracy
            prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1, 5))
            print(f"Prec@1: {prec1}, Prec@5: {prec5}")
            top1.update(prec1, audio_inputs.size(0))
            top5.update(prec5, audio_inputs.size(0))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        except Exception as e:
            print(f"Error in batch {i}: {str(e)}")

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # Logging batch information with a check on .item()
        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val.item() if isinstance(losses.val, torch.Tensor) else losses.val,
            'prec1': top1.val.item() if isinstance(top1.val, torch.Tensor) else top1.val,
            'prec5': top5.val.item() if isinstance(top5.val, torch.Tensor) else top5.val,
            'lr': optimizer.param_groups[0]['lr']
        })

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                epoch,
                i,
                len(data_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                top1=top1,
                top5=top5,
                lr=optimizer.param_groups[0]['lr']))

    # Logging epoch information with a check on .item()
    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.item() if isinstance(losses.avg, torch.Tensor) else losses.avg,
        'prec1': top1.avg.item() if isinstance(top1.avg, torch.Tensor) else top1.avg,
        'prec5': top5.avg.item() if isinstance(top5.avg, torch.Tensor) else top5.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

    # Update the lists with the average loss and accuracy for the epoch
    train_losses.append(losses.avg.item() if isinstance(losses.avg, torch.Tensor) else losses.avg)
    train_prec1s.append(top1.avg.item() if isinstance(top1.avg, torch.Tensor) else top1.avg)


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger, train_losses, train_prec1s):
    print('train at epoch {}'.format(epoch))

    if opt.model == 'multimodalcnn':
        print("Iniziare l'addestramento multimodale")
        train_epoch_multimodal(epoch, data_loader, model, criterion, optimizer, opt, epoch_logger, batch_logger,
                               train_losses, train_prec1s)
    else:
        print("Modello non riconosciuto, eseguire il codice per il modello specificato.")

    return epoch_logger, batch_logger  # Assicurati di restituire sempre i logger
