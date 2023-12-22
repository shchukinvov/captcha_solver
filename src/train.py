from captcha_solver.config import TRAIN_IMAGE_DIR, TEST_IMAGE_DIR
from models import CaptchaSolverLSTM, CaptchaSolverAtt
from dataset import init_dataloaders
from loss_fn import CaptchaCTCLoss
from utils import *
from transforms import TRAIN_TRANSFORMS, VAL_TRANSFORMS


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def main():
    seed_all(1)
    model = CaptchaSolverLSTM(num_chars=len(CHAR_LIST)).to(DEVICE)
    model.load_state_dict(torch.load("../models/model2.pth")['state_dict'])
    criterion = CaptchaCTCLoss().to(DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-4)
    optimizer.load_state_dict(torch.load("../models/model1.pth")['optimizer'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    train_loader, val_loader, test_loader = init_dataloaders(
        image_dir=TRAIN_IMAGE_DIR,
        test_image_dir=TEST_IMAGE_DIR,
        train_transforms=TRAIN_TRANSFORMS,
        val_transforms=VAL_TRANSFORMS,
        batch_size=32,
        rand_state=1,
        num_workers=2,
    )
    train_loss, val_loss, cer_scores, acc_scores = train_fn(
        model=model,
        dataloaders=(train_loader, val_loader),
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=20,
        device=DEVICE,
        scheduler=scheduler
    )
    torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, "../models/model2.pth")
    show_losses(train_loss, val_loss)
    test_cer, test_acc = validate_func(model, test_loader)
    print(test_cer, test_acc)


if __name__ == "__main__":
    main()
