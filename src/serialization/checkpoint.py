import pickle

# How to use:
# model = Sequential(...)
#
# model.compile(
#     optimizer="adam",
#     loss="cross_entropy",
# )
#
# epoch, loss = Checkpoint.load(
#     model,
#     model.optimizer,
#     "checkpoint.ptf",
# )
#
# model.fit(
#     X_train,
#     y_train,
#     epochs=100,
#     initial_epoch=epoch,
# )

class Checkpoint:

    @staticmethod
    def save(
        model,
        optimizer,
        epoch,
        loss,
        path,
    ):

        state = {
            "epoch": epoch,
            "loss": float(loss.data),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

    @staticmethod
    def load(
        model,
        optimizer,
        path,
    ):

        with open(path, "rb") as f:
            state = pickle.load(f)

        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])

        return state["epoch"], state["loss"]