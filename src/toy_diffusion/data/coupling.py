import torch
from torch.utils.data import Dataset
from toy_diffusion.paths.sampling import generate_samples


class Coupling(Dataset):
    """
    Dataset for Rectified Flow (Reflow).

    It generates and stores pairs of (Noise z0, Generated Data z1) using a
    pretrained Flow Matching model.

    Training on these pairs straightens the ODE trajectory because the new
    target velocity (z1 - z0) represents the straight line connecting the
    noise to the generated sample.
    """

    def __init__(
        self,
        model,
        schedule,
        num_samples,
        dim=2,
        device="cuda",
        batch_size=2048,
        autocast_dtype=torch.float32,
        clip_prediction: bool = False,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.autocast_enabled = False if autocast_dtype == torch.float32 else True

        # int (flat) or tuple (image)
        if isinstance(dim, int):
            self.data_shape = (dim,)
        else:
            self.data_shape = dim

        self.z0, self.z1 = self._generate_data(
            model, schedule, batch_size, clip_prediction=clip_prediction
        )

    def _generate_data(self, model, schedule, batch_size, clip_prediction=False):
        print(f"Generating Coupling Data ({self.num_samples} pairs)...")
        schedule_name = schedule.get_scheduler_type()
        z0_list = []
        z1_list = []

        if isinstance(model, dict) and not isinstance(model, torch.nn.ModuleDict):
            for m in model.values():
                m.eval()
        else:
            model.eval()

        device_type_str = torch.device(self.device).type
        is_conditional = (
            isinstance(model, (dict, torch.nn.ModuleDict)) and "text_enc" in model
        )

        num_batches = (self.num_samples + batch_size - 1) // batch_size
        with torch.autocast(
            device_type=device_type_str,
            dtype=self.autocast_dtype,
            enabled=self.autocast_enabled,
        ):
            with torch.no_grad():
                for _ in range(num_batches):
                    current_bs = min(
                        batch_size, self.num_samples - len(z0_list) * batch_size
                    )

                    # 1. Sample Noise z0
                    z0 = torch.randn(current_bs, *self.data_shape, device=self.device)

                    cond_embeds = None
                    if is_conditional:
                        cond_embeds = model["text_enc"]([""] * current_bs)

                    prediction_target = "v" if schedule_name == "linear" else "eps"

                    # 2. Solve ODE to get z1
                    extra_kwargs = {}
                    if schedule_name == "ddpm":
                        extra_kwargs["clip_prediction"] = clip_prediction

                    z1 = generate_samples(
                        model=model,
                        schedule=schedule,
                        x=z0.clone(),
                        diffusion_type=schedule_name,
                        prediction_target=prediction_target,
                        num_steps=50,
                        is_conditional=is_conditional,
                        cond_embeds=cond_embeds,
                        projection_matrix=None,
                        return_traj=False,
                        **extra_kwargs,
                    )

                    z0_list.append(z0.float().cpu())
                    z1_list.append(torch.from_numpy(z1).cpu())

        z0_all = torch.cat(z0_list, dim=0)[: self.num_samples]
        z1_all = torch.cat(z1_list, dim=0)[: self.num_samples]

        print(f"Coupling generated. Shape: {z0_all.shape}")
        return z0_all, z1_all

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return (Noise, Data) pair
        return self.z0[idx], self.z1[idx]
