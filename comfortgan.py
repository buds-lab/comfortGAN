from utils import *

# PyTorch
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Generator(nn.Module):
    def __init__(self, latent_dim, data_dim, n_classes):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.data_dim = data_dim
        
        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.BatchNorm1d(out_feat)) # layers.append(nn.BatchNorm1d(out_feat, 0.8)) # OG
            layers.append(nn.ReLU()) # layers.append(nn.LeakyReLU(0.2, inplace=True)) # OG
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim + self.n_classes, 128), # *block(opt.latent_dim + opt.n_classes, 128, normalize=False), #OG
            *block(128, 256),
            *block(256, 128),
            *block(128, 64),
            *block(64, 32),
            nn.Linear(32, self.data_dim),
            nn.ReLU()
        )

    def forward(self, noise, labels_ohe_noisy): # G(Z)
        # Concatenate label and data to produce input
        gen_in = torch.cat((labels_ohe_noisy, noise), -1)
        data_gen = self.model(gen_in)
        return data_gen

class Discriminator(nn.Module):
    def __init__(self, data_dim, n_classes):
        super(Discriminator, self).__init__()
        self.n_classes = n_classes
        self.data_dim = data_dim
        
        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.Dropout(0.5))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(self.data_dim + self.n_classes, 64),
            *block(64, 128),
            *block(128, 64),
            *block(64, 32),
            *block(32, 16),
            nn.Linear(16, 1)
        )

    def forward(self, data, labels_ohe):
        # Concatenate label and data to produce input
        disc_in = torch.cat((data, labels_ohe), -1)
        validity = self.model(disc_in)
        return validity

class comfortGAN(object):
    """
    Conditional Wassertein GAN
    Args:
            num_cont (int):
                number of continous columns in original dataset
            scaler (Scaler object):
                continous columns scaler
            ohe_cat (Scaler object):
                categorical columns scaler
            ohe_y (Scaler object):
                label column scaler
            columns_names (list of string):
                name of original dataset columns
            cat_cols_idx (list of int):
                indices of the categorical columns
    """
    def __init__(self, data_dim, latent_dim, gamma, list_label,
                num_cont, columns_names, cat_cols_idx, scaler, ohe_cat, ohe_label, 
                lr=2e-4, lambda_gp=10, beta_1=0.5, beta_2=0.9):

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.list_label = list_label
        self.num_cont = num_cont
        self.columns_names = columns_names
        self.cat_cols_idx = cat_cols_idx
        self.scaler = scaler
        self.ohe_cat = ohe_cat
        self.ohe_label = ohe_label
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.lambda_gp = lambda_gp
        self.num_classes = len(list_label)
        self.min_class = np.amin(list_label)
        self.max_class = np.amax(list_label)
        self.cuda = True if torch.cuda.is_available() else False
        self.device = "cuda" if self.cuda else "cpu"
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        # Initialize generator and discriminator
        self.generator = Generator(self.latent_dim, self.data_dim, self.num_classes).to(self.device)
        self.discriminator = Discriminator(self.data_dim, self.num_classes).to(self.device)
        
        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta_1, self.beta_2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta_1, self.beta_2))
    
        # print network architecture
        self.print_networks()

    def print_networks(self):
        print_network(self.generator)
        print_network(self.discriminator)

    def compute_gradient_penalty(self, real_samples, real_labels, gen_samples):
        """
        Calculates the gradient penalty loss for WGAN GP
        Args:
            real_samples (torch.Tensor):
                instances from the original dataset
            fake_samples (torch.Tensor):
                instances sampled from the latent space
        Return:
            gradient_penalty (int):
                penalty value
        """
        
        # interpolation of real and generated samples
        alpha = torch.rand(real_samples.size(0), 1, device=self.device) # random weight term - Uniform[0, 1]

        interpolates = (alpha * real_samples + ((1 - alpha) * gen_samples)).requires_grad_(True) # x_hat
        disc_interpolates = self.discriminator(interpolates, real_labels)

        # gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp
        
        return gradient_penalty

    def sample_latent_space(self, sample_size, sample=False):
        # sample latent space
        mean_z = torch.zeros((sample_size, self.latent_dim), device=self.device)
        std_z = torch.ones((sample_size, self.latent_dim), device=self.device)
        z = self.Tensor(torch.normal(mean=mean_z, std=std_z))

        # generate random labels
        # since randint high parameter is exclusive, add 1
        gen_y = self.Tensor(np.random.randint(self.min_class, self.max_class + 1, sample_size))
        if self.cuda:
            gen_y = gen_y.cpu()
        gen_y_oh = self.Tensor(self.ohe_label.transform(gen_y.reshape(-1, 1))) # one-hot encode
        # adds uniform noise to y-vector
        a = torch.zeros(gen_y_oh.shape, device=self.device)
        b = torch.ones(gen_y_oh.shape, device=self.device) * self.gamma
        noise_y = self.Tensor(torch.distributions.Uniform(a, b).sample())
        norm_denominator = torch.sum(gen_y_oh + noise_y, dim=1)
        norm_denominator = torch.unsqueeze(norm_denominator, 1)
        gen_y_noise = torch.div(gen_y_oh + noise_y, norm_denominator) # normalize

        if sample: 
            return z, gen_y_noise, gen_y
        else:
            return z

    def train(self, data_train, epochs, n_critic=5, run_name='default', log=True):
        """
        Trains GAN architecture.
        Args:
            data_train (iterable dataset):
                batche-sliced dataset from DataLoader, includes features and labels
            epochs (int):
                number of epochs for the training process
            n_critic (int):
                number of critics for WGAN per generator iteration
            run_name (String):
                name for tensorboard folder
            log (boolean):
                whether to print loss information during training process
        """
        
        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        
        writer = SummaryWriter('tensorboard/' + run_name + "T" + current_time)
        n_steps = len(list(data_train))
        prev_step = 0
        
        ###########
        #  Training
        ###########
        g_loss_steps = []
        d_loss_steps = []
        
        for epoch in range(epochs):
            g_loss_epochs = []
            d_loss_epochs = []
            
            for step, batch_train in enumerate(data_train):
                step = prev_step + step # tensorboard iteration count
                curr_batch_size = batch_train.numpy().shape[0]
                
                # get training features and their labels
                # since it's one-hot-encoded, there are as many columns as number of classes    
                y_batch = self.Tensor(batch_train[:, -self.num_classes:].numpy())
                X_batch = self.Tensor(batch_train[:, 0:-self.num_classes].numpy()) # remaining columns are features
                
                ######################
                #  Train Discriminator
                ######################
                z = self.sample_latent_space(curr_batch_size)
                
                # generate a batch from latent space BUT with the original labels
                gen_X = self.generator(z, y_batch)
                
                # forward pass over discriminator
                d_real = self.discriminator(X_batch, y_batch)
                d_gen = self.discriminator(gen_X.detach(), y_batch)

                # loss
                gradient_penalty = self.compute_gradient_penalty(X_batch, y_batch, gen_X)
                d_loss = torch.mean(d_gen) - torch.mean(d_real) + gradient_penalty
    
                self.optimizer_D.zero_grad()
                d_loss.backward()
                self.optimizer_D.step()
                
                if step % n_critic == 0: # train the generator every n_critic steps
                    ##################
                    #  Train Generator
                    ##################
                    z = self.sample_latent_space(curr_batch_size)

                    # generate a batch from latent space
                    gen_X = self.generator(z, y_batch)
                
                    # forward pass over discriminator
                    d_gen = self.discriminator(gen_X.detach(), y_batch)
                    g_loss = -torch.mean(d_gen)
                    
                    self.optimizer_G.zero_grad()
                    g_loss.backward()
                    self.optimizer_G.step()
                            
                    ######################
                    # Keep track of losses
                    ######################
                    writer.add_scalars(run_name , {
                        'gen_loss': g_loss.item(),
                        'disc_loss': d_loss.item(),
                    }, step)
                    step += 1
                    g_loss_steps.append(g_loss.item())
                    d_loss_steps.append(d_loss.item())
                    g_loss_epochs.append(g_loss.item())
                    d_loss_epochs.append(d_loss.item())
            
            # print losses every 1000 iterations
            if log and ((epoch + 1) % 1000 == 0):
                print("[Epoch %d/%d] [D loss: %f] [G loss: %f]" 
                    % (epoch + 1, epochs, np.mean(d_loss_epochs), np.mean(g_loss_epochs))
                    )
            prev_step = step

        return g_loss_steps, d_loss_steps

    def generate_batch(self, sample_size):
        # generate a batch from latent space
        z, gen_y_noise, gen_y = self.sample_latent_space(sample_size, sample=True)
        gen_X = self.generator(z, gen_y_noise)
                
        # rescale back generated samples (continous)
        if self.cuda:
            gen_X_np = gen_X.cpu().detach().numpy()
        else:
            gen_X_np = gen_X.detach().numpy()
        gen_X_cont = self.scaler.inverse_transform(gen_X_np[:, 0:self.num_cont])
        # rescale categorical only if the dataset has any
        if self.ohe_cat is not None:
            gen_X_cat = self.ohe_cat.inverse_transform(gen_X_np[:, self.num_cont:])
        
        # concatenate continous and label
        gen_X_y = np.concatenate((gen_X_cont, gen_y.reshape(-1, 1)), axis=1)
        
        # if the dataset has categorical columns, insert them
        # in their original order
        if self.ohe_cat is not None:
            idx = 0
            for cat_col in self.cat_cols_idx:
                gen_X_y = np.insert(gen_X_y, cat_col, gen_X_cat[:, idx], axis=1)
                idx += 1
        
        df_gen = pd.DataFrame(gen_X_y, columns=self.columns_names)
        
        return df_gen

def find_hyperparam(data_train, cat_cols, scaling, gamma_list, n_critic_list, BATCH_SIZE_list):
    """
    Train multiple GANs with different values of hyperparameters.
    Args:
        data_train (pandas.DataFrame):
            datframe with all features and label
        cat_cols (list):
            list of indices of the categorical columns in data_train
        scaling (string):
            type of scaling for continuous features
        gamma_list (list):
            intended values to try for gamma
        BATCH_SIZE_list (list):
            intended values to try for BATCH_SIZE
        n_critic_list (list):
            list of values for number of critics for WGAN
    """
    ##################
    # Fixed parameters
    ##################
    seed = 13 
    columns_names = data_train.columns.values

    for BATCH_SIZE in BATCH_SIZE_list:
        # in order to have around 20k iterations for, we solve for the number of epochs
        EPOCHS = max(20000 * BATCH_SIZE // len(data_train), 1)
        print("EPOCHS: {}".format(EPOCHS))

        for n_critic in n_critic_list:
            for gamma in gamma_list:
                # transform data
                X_encoded, scaler, ohe_cat, ohe_label, list_label, num_cont = data_transform(data_train, cat_cols=cat_cols, scaling=scaling, gamma=gamma)
                X_train = torch.utils.data.DataLoader(X_encoded, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=seed)
                data_dim = X_encoded.shape[1] - len(list_label) # susbract the columns from the one-hot-encoded label

                latent_dim_list = [data_dim * 2, data_dim * 4, data_dim * 8, 100, 150]

                for latent_dim in latent_dim_list:
                    run_name = str(BATCH_SIZE) + "-" + str(n_critic) +  "-" + str(gamma) + "-" + str(latent_dim)
                    print("Running Experiment with BATCH_SIZE: {}, n_critic: {}, gama: {}, latent_dim: {}".format(BATCH_SIZE, n_critic, gamma, latent_dim))
                    
                    # initialize gan
                    comfortgan = comfortGAN(data_dim, 
                                            latent_dim, 
                                            gamma, 
                                            list_label, 
                                            num_cont, 
                                            columns_names, 
                                            cat_cols, 
                                            scaler, 
                                            ohe_cat, 
                                            ohe_label)
                
                    comfortgan.train(X_train, EPOCHS, n_critic, run_name='default', log=True)
