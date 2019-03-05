import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channel, kernel_height, kernel_width):
        super(ConvUnit, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channel,
                               kernel_size=(kernel_height, kernel_width),
                               stride=1,
                               bias=True)

    def forward(self, x):
        return self.conv0(x)


class CapsuleLayer(nn.Module):
    def __init__(self, in_units, in_channels, num_units, unit_size, use_routing,
                 conv_unit_out_channel, conv_unit_kernel_height, conv_unit_kernel_width):
        super(CapsuleLayer, self).__init__()

        self.in_units = in_units
        self.in_channels = in_channels
        self.num_units = num_units
        self.use_routing = use_routing

        if self.use_routing:
            # In the paper, the deeper capsule layer(s) with capsule inputs (DigitCaps) use a special routing algorithm
            # that uses this weight matrix.
            self.W = nn.Parameter(torch.randn(1, in_channels, num_units, unit_size, in_units))
        else:
            # The first convolutional capsule layer (PrimaryCapsules in the paper) does not perform routing.
            # Instead, it is composed of several convolutional units, each of which sees the full input.
            # It is implemented as a normal convolutional layer with a special nonlinearity (squash()).
            def create_conv_unit(unit_idx):
                unit = ConvUnit(in_channels, conv_unit_out_channel,
                                conv_unit_kernel_height, conv_unit_kernel_width)
                self.add_module("unit_" + str(unit_idx), unit)
                return unit
            self.units = [create_conv_unit(i) for i in range(self.num_units)]

    @staticmethod
    def squash(s):
        # This is equation 1 from the paper.
        mag_sq = torch.sum(s**2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, x):
        if self.use_routing:
            return self.routing(x)
        else:
            return self.no_routing(x)

    def no_routing(self, x):
        # Get output for each unit.
        # Each will be (batch, channels, height, width).
        u = [self.units[i](x) for i in range(self.num_units)]

        # Stack all unit outputs (batch, unit, channels, height, width).
        u = torch.stack(u, dim=1)

        # Flatten to (batch, unit, output).
        u = u.view(x.size(0), self.num_units, -1)

        # Return squashed outputs.
        return CapsuleLayer.squash(u)

    def routing(self, x):
        batch_size = x.size(0)

        # (batch, in_units, features) -> (batch, features, in_units)
        x = x.transpose(1, 2)

        # (batch, features, in_units) -> (batch, features, num_units, in_units, 1)
        x = torch.stack([x] * self.num_units, dim=2).unsqueeze(4)

        # (batch, features, in_units, unit_size, num_units)
        W = torch.cat([self.W] * batch_size, dim=0)

        # Transform inputs by weight matrix.
        # (batch_size, features, num_units, unit_size, 1)
        u_hat = torch.matmul(W, x)

        # Initialize routing logits to zero.
        b_ij = Variable(torch.zeros(1, self.in_channels, self.num_units, 1))
        if torch.cuda.is_available():
            b_ij = b_ij.cuda()

        # Iterative routing.
        num_iterations = 3
        for iteration in range(num_iterations):
            # Convert routing logits to softmax.
            # (batch, features, num_units, 1, 1)
            c_ij = F.softmax(b_ij)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            # Apply routing (c_ij) to weighted inputs (u_hat).
            # (batch_size, 1, num_units, unit_size, 1)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            # (batch_size, 1, num_units, unit_size, 1)
            v_j = CapsuleLayer.squash(s_j)

            # (batch_size, features, num_units, unit_size, 1)
            v_j1 = torch.cat([v_j] * self.in_channels, dim=1)

            # (1, features, num_units, 1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)

            # Update b_ij (routing)
            b_ij = b_ij + u_vj1

        return v_j.squeeze(1)


class CapsuleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_height, kernel_width):
        super(CapsuleConvLayer, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(kernel_height, kernel_width),
                               stride=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv0(x))


class CapsuleNetwork(nn.Module):
    def __init__(self,
                 image_width, image_height, image_channels,
                 conv_outputs, conv_kernel_height, conv_kernel_width,
                 num_primary_units, primary_unit_size, conv_unit_out_channel, conv_unit_kernel_height, conv_unit_kernel_width,
                 num_output_units, output_unit_size):
        """
        胶囊网络

        ***** 第一层卷积的参数 *****
        :param image_width:
        :param image_height:
        :param image_channels:
        :param conv_outputs: 第一个卷积层的输出channel个数
        :param conv1_kernel_height: 第一个卷积层的卷积核的高度
        :param conv1_kernel_width: 第一个卷积层的卷积核的宽度

        ***** primary层的参数 *****
        :param num_primary_units: primary层算出来的每个胶囊的维度
        :param primary_unit_size: primary层算出来的胶囊的个数
        :param conv_unit_out_channel: primary层中小卷积单元的输出channel个数
        :param conv_unit_kernel_height: primary层中小卷积单元的卷积核高度
        :param conv_unit_kernel_width: primary层中小卷积单元的卷积核宽度

        ***** 胶囊层的参数 *****
        :param num_output_units: digits层算出来的每个胶囊的维度
        :param output_unit_size: digits层算出来的胶囊的个数
        """
        super(CapsuleNetwork, self).__init__()

        self.reconstructed_image_count = 0

        self.image_channels = image_channels
        self.image_width = image_width
        self.image_height = image_height

        self.conv1 = CapsuleConvLayer(in_channels=image_channels,
                                      out_channels=conv_outputs,
                                      kernel_height=conv_kernel_height,
                                      kernel_width=conv_kernel_width)

        self.primary = CapsuleLayer(in_units=0,
                                    in_channels=conv_outputs,
                                    num_units=num_primary_units,
                                    unit_size=primary_unit_size,
                                    use_routing=False,
                                    conv_unit_out_channel=conv_unit_out_channel,
                                    conv_unit_kernel_height=conv_unit_kernel_height,
                                    conv_unit_kernel_width=conv_unit_kernel_width)

        self.digits = CapsuleLayer(in_units=num_primary_units,
                                   in_channels=primary_unit_size,
                                   num_units=num_output_units,
                                   unit_size=output_unit_size,
                                   use_routing=True,
                                   conv_unit_out_channel=conv_unit_out_channel,
                                   conv_unit_kernel_height=conv_unit_kernel_height,
                                   conv_unit_kernel_width=conv_unit_kernel_width)

        reconstruction_size = image_width * image_height * image_channels
        self.reconstruct0 = nn.Linear(num_output_units*output_unit_size, int((reconstruction_size * 2) / 3))
        self.reconstruct1 = nn.Linear(int((reconstruction_size * 2) / 3), int((reconstruction_size * 3) / 2))
        self.reconstruct2 = nn.Linear(int((reconstruction_size * 3) / 2), reconstruction_size)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        :param x:  [batch_size, 1, x_height, x_width]
        :return: [batch_size, ]
        """
        conv1_out = self.conv1(x)  # 维度: [
        #   batch_size,
        #   conv_outputs,
        #   conv1_out_h = x_height - conv1_kernel_height + 1,
        #   conv1_out_w = x_width - conv1_kernel_width + 1
        # ]
        primary_out = self.primary(conv1_out)  # 维度: [
        #   batch_size,
        #   num_primary_units,
        #   conv_unit_out_channel * (conv1_out_h-conv_unit_kernel_height+1) * (conv1_out_w-conv_unit_kernel_width+1)
        # ]
        digits_out = self.digits(primary_out)  # [batch_size, num_output_units, output_unit_size, 1]
        return digits_out

    def loss(self, images, input, target, size_average=True):
        return self.margin_loss(input, target, size_average) + self.reconstruction_loss(images, input, size_average)

    def margin_loss(self, input, target, size_average=True):
        batch_size = input.size(0)

        # ||vc|| from the paper.
        v_mag = torch.sqrt((input**2).sum(dim=2, keepdim=True))

        # Calculate left and right max() terms from equation 4 in the paper.
        zero = Variable(torch.zeros(1))
        if torch.cuda.is_available():
            zero = zero.cuda()
        m_plus = 0.9
        m_minus = 0.1
        max_l = torch.max(m_plus - v_mag, zero).view(batch_size, -1)**2
        max_r = torch.max(v_mag - m_minus, zero).view(batch_size, -1)**2

        # This is equation 4 from the paper.
        loss_lambda = 0.5
        T_c = target
        L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
        L_c = L_c.sum(dim=1)

        if size_average:
            L_c = L_c.mean()

        return L_c

    def reconstruction_loss(self, images, input, size_average=True):
        # Get the lengths of capsule outputs.
        v_mag = torch.sqrt((input**2).sum(dim=2))

        # Get index of longest capsule output.
        _, v_max_index = v_mag.max(dim=1)
        v_max_index = v_max_index.data

        # Use just the winning capsule's representation (and zeros for other capsules) to reconstruct input image.
        batch_size = input.size(0)
        all_masked = [None] * batch_size
        for batch_idx in range(batch_size):
            # Get one sample from the batch.
            input_batch = input[batch_idx]

            # Copy only the maximum capsule index from this batch sample.
            # This masks out (leaves as zero) the other capsules in this sample.
            batch_masked = Variable(torch.zeros(input_batch.size()))
            if torch.cuda.is_available():
                batch_masked = batch_masked.cuda()
            batch_masked[v_max_index[batch_idx]] = input_batch[v_max_index[batch_idx]]
            all_masked[batch_idx] = batch_masked

        # Stack masked capsules over the batch dimension.
        masked = torch.stack(all_masked, dim=0)

        # Reconstruct input image.
        masked = masked.view(input.size(0), -1)
        output = self.relu(self.reconstruct0(masked))
        output = self.relu(self.reconstruct1(output))
        output = self.sigmoid(self.reconstruct2(output))
        output = output.view(-1, self.image_channels, self.image_height, self.image_width)

        # Save reconstructed images occasionally.
        if self.reconstructed_image_count % 10 == 0:
            if output.size(1) == 2:
                # handle two-channel images
                zeros = torch.zeros(output.size(0), 1, output.size(2), output.size(3))
                output_image = torch.cat([zeros, output.data.cpu()], dim=1)
            else:
                # assume RGB or grayscale
                output_image = output.data.cpu()
        self.reconstructed_image_count += 1

        # The reconstruction loss is the sum squared difference between the input image and reconstructed image.
        # Multiplied by a small number so it doesn't dominate the margin (class) loss.
        error = (output - images).view(output.size(0), -1)
        error = error**2
        error = torch.sum(error, dim=1) * 0.0005

        # Average over batch
        if size_average:
            error = error.mean()

        return error