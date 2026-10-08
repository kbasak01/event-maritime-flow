# ruff: noqa
# Vendored verbatim from tudelft/event_flow (https://github.com/tudelft/event_flow),
# commit e81f9630a145da8c7e57b2cfd38cf6c0b47ac14f, for use as a TEST ORACLE ONLY.
# Never imported by src/ (tests/test_oracle_parity.py checks this). Listed in docs/LICENSES.md.
#
# Contents, unmodified except as noted:
#   utils/iwe.py   lines 4-92:   purge_unfeasible, get_interpolation, interpolate
#   loss/flow.py   lines 13-23:  spatial_variance
#   loss/flow.py   lines 304-628: BaseValidationLoss, FWL, RSAT, AEE
# Modification: the sys.path hack and "from utils.iwe import ..." of loss/flow.py are replaced
# by defining those functions in this file.
#
# MIT License
#
# Copyright (c) 2021 TU Delft
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch


def purge_unfeasible(x, res):
    """
    Purge unfeasible event locations by setting their interpolation weights to zero.
    :param x: location of motion compensated events
    :param res: resolution of the image space
    :return masked indices
    :return mask for interpolation weights
    """

    mask = torch.ones((x.shape[0], x.shape[1], 1)).to(x.device)
    mask_y = (x[:, :, 0:1] < 0) + (x[:, :, 0:1] >= res[0])
    mask_x = (x[:, :, 1:2] < 0) + (x[:, :, 1:2] >= res[1])
    mask[mask_y + mask_x] = 0
    return x * mask, mask


def get_interpolation(events, flow, tref, res, flow_scaling, round_idx=False):
    """
    Warp the input events according to the provided optical flow map and compute the bilinar interpolation
    (or rounding) weights to distribute the events to the closes (integer) locations in the image space.
    :param events: [batch_size x N x 4] input events (ts, y, x, p)
    :param flow: [batch_size x N x 2] optical flows (y, x)
    :param tref: reference time toward which events are warped
    :param res: resolution of the image space
    :param flow_scaling: scalar that multiplies the optical flow map
    :param round_idx: whether or not to round the event locations instead of doing bilinear interp. (default = False)
    :return interpolated event indices
    :return interpolation weights
    """

    # event propagation
    # flow scaling: max flow is largest dimension, combined with tanh this gets flow across entire sensor
    # (maybe not necessary for linear activation)
    warped_events = events[:, :, 1:3] + (tref - events[:, :, 0:1]) * flow * flow_scaling

    if round_idx:

        # no bilinear interpolation
        idx = torch.round(warped_events)
        weights = torch.ones(idx.shape).to(events.device)

    else:

        # get scattering indices
        top_y = torch.floor(warped_events[:, :, 0:1])
        bot_y = torch.floor(warped_events[:, :, 0:1] + 1)
        left_x = torch.floor(warped_events[:, :, 1:2])
        right_x = torch.floor(warped_events[:, :, 1:2] + 1)

        top_left = torch.cat([top_y, left_x], dim=2)
        top_right = torch.cat([top_y, right_x], dim=2)
        bottom_left = torch.cat([bot_y, left_x], dim=2)
        bottom_right = torch.cat([bot_y, right_x], dim=2)
        idx = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=1)

        # get scattering interpolation weights
        warped_events = torch.cat([warped_events for i in range(4)], dim=1)
        zeros = torch.zeros(warped_events.shape).to(events.device)
        weights = torch.max(zeros, 1 - torch.abs(warped_events - idx))

    # purge unfeasible indices
    idx, mask = purge_unfeasible(idx, res)

    # make unfeasible weights zero
    weights = torch.prod(weights, dim=-1, keepdim=True) * mask  # bilinear interpolation

    # prepare indices
    idx[:, :, 0] *= res[1]  # torch.view is row-major
    idx = torch.sum(idx, dim=2, keepdim=True)

    return idx, weights


def interpolate(idx, weights, res, polarity_mask=None):
    """
    Create an image-like representation of the warped events.
    :param idx: [batch_size x N x 1] warped event locations
    :param weights: [batch_size x N x 1] interpolation weights for the warped events
    :param res: resolution of the image space
    :param polarity_mask: [batch_size x N x 1] polarity mask for the warped events (default = None)
    :return image of warped events
    """

    if polarity_mask is not None:
        weights = weights * polarity_mask
    iwe = torch.zeros((idx.shape[0], res[0] * res[1], 1)).to(idx.device)
    iwe = iwe.scatter_add_(1, idx.long(), weights)
    iwe = iwe.view((idx.shape[0], 1, res[0], res[1]))
    return iwe


def spatial_variance(x):
    return torch.var(
        x.view(
            x.shape[0],
            x.shape[1],
            1,
            -1,
        ),
        dim=3,
        keepdim=True,
    )


class BaseValidationLoss(torch.nn.Module):
    """
    Base class for validation metrics.
    """

    def __init__(self, config, device, flow_scaling=128):
        super(BaseValidationLoss, self).__init__()
        self.res = config["loader"]["resolution"]
        self.flow_scaling = flow_scaling  # should be specified by the user
        self.overwrite_intermediate = (
            False if "overwrite_intermediate" not in config["loss"].keys() else config["loss"]["overwrite_intermediate"]
        )
        self.device = device

        self._passes = 0
        self._event_list = None
        self._flow_list = None
        self._flow_map = None
        self._pol_mask_list = None
        self._event_mask = None

    @property
    def num_events(self):
        if self._event_list is None:
            return 0
        else:
            return self._event_list.shape[1]

    def event_flow_association(self, flow_list, inputs):
        """
        :param flow_list: [[batch_size x 2 x H x W]] list of optical flow (x, y) maps
        :param inputs: dataloader dictionary
        """

        # move to device
        event_list = inputs["event_list"].to(self.device)
        pol_mask = inputs["event_list_pol_mask"].to(self.device)
        event_mask = inputs["event_mask"].to(self.device)
        gtflow = inputs["gtflow"].to(self.device) if "gtflow" in inputs.keys() else None

        # flow vector per input event
        flow_idx = event_list[:, :, 1:3].clone()
        flow_idx[:, :, 0] *= self.res[1]  # torch.view is row-major
        flow_idx = torch.sum(flow_idx, dim=2)

        # get flow for every event in the list
        flow = flow_list[-1]  # only highest resolution flow
        flow = flow.view(flow.shape[0], 2, -1)
        event_flowy = torch.gather(flow[:, 1, :], 1, flow_idx.long())  # vertical component
        event_flowx = torch.gather(flow[:, 0, :], 1, flow_idx.long())  # horizontal component
        event_flowy = event_flowy.view(event_flowy.shape[0], event_flowy.shape[1], 1)
        event_flowx = event_flowx.view(event_flowx.shape[0], event_flowx.shape[1], 1)
        event_flow = torch.cat([event_flowy, event_flowx], dim=2)

        if self._flow_list is None:
            self._flow_list = event_flow
        else:
            self._flow_list = torch.cat([self._flow_list, event_flow], dim=1)

        # update internal event list
        if self._event_list is None:
            self._event_list = event_list
        else:
            event_list = event_list.clone()  # to prevent issues with other metrics
            event_list[:, :, 0:1] += self._passes  # only nonzero second time
            self._event_list = torch.cat([self._event_list, event_list], dim=1)

        # update internal polarity mask list
        if self._pol_mask_list is None:
            self._pol_mask_list = pol_mask
        else:
            self._pol_mask_list = torch.cat([self._pol_mask_list, pol_mask], dim=1)

        # update flow map
        if self._flow_map is None:
            self._flow_map = []
        self._flow_map.append(flow.view(flow.shape[0], 2, self.res[0], self.res[1]))

        # update ground-truth optical flow
        self._gtflow = gtflow

        # event mask
        if self._event_mask is None:
            self._event_mask = event_mask
        else:
            self._event_mask = torch.cat([self._event_mask, event_mask], dim=1)

        # update timestamps
        self._dt_input = inputs["dt_input"]
        self._dt_gt = inputs["dt_gt"]

        # update timestamp index
        self._passes += 1

    def overwrite_intermediate_flow(self, flow_list):
        """
        :param flow_list: [[batch_size x 2 x H x W]] list of optical flow (x, y) maps
        """

        # flow vector per input event
        flow_idx = self._event_list[:, :, 1:3].clone()
        flow_idx[:, :, 0] *= self.res[1]  # torch.view is row-major
        flow_idx = torch.sum(flow_idx, dim=2)

        # get flow for every event in the list
        flow = flow_list[-1]  # only highest resolution flow
        flow = flow.view(flow.shape[0], 2, -1)
        event_flowy = torch.gather(flow[:, 1, :], 1, flow_idx.long())  # vertical component
        event_flowx = torch.gather(flow[:, 0, :], 1, flow_idx.long())  # horizontal component
        event_flowy = event_flowy.view(event_flowy.shape[0], event_flowy.shape[1], 1)
        event_flowx = event_flowx.view(event_flowx.shape[0], event_flowx.shape[1], 1)
        event_flow = torch.cat([event_flowy, event_flowx], dim=2)

        self._flow_list = event_flow
        self._flow_map = [flow.view(flow.shape[0], 2, self.res[0], self.res[1])]

        # update mask
        self._event_mask = torch.sum(self._event_mask, dim=1, keepdim=True)
        self._event_mask[self._event_mask > 1] = 1

    def reset(self):
        self._passes = 0
        self._event_list = None
        self._flow_list = None
        self._flow_map = None
        self._pol_mask_list = None
        self._event_mask = None

    def compute_window_events(self):
        idx = self._event_list[:, :, 1:3].clone()
        idx[:, :, 0] *= self.res[1]  # torch.view is row-major
        idx = torch.sum(idx, dim=2, keepdim=True)
        weights = torch.ones(idx.shape).to(self.device)

        events_pos = interpolate(idx.long(), weights, self.res, polarity_mask=self._pol_mask_list[:, :, 0:1])
        events_neg = interpolate(idx.long(), weights, self.res, polarity_mask=self._pol_mask_list[:, :, 1:2])

        return torch.cat([events_pos, events_neg], dim=1)

    def compute_masked_window_flow(self):

        if self.overwrite_intermediate:
            return self._flow_map[-1] * self._event_mask
        else:
            avg_flow = self._flow_map[0] * self._event_mask[:, 0:1, :, :]
            for i in range(1, self._event_mask.shape[1]):
                avg_flow += self._flow_map[i] * self._event_mask[:, i : i + 1, :, :]
            avg_flow /= torch.sum(self._event_mask, dim=1, keepdim=True) + 1e-9
            return avg_flow

    def compute_window_iwe(self, round_idx=True):
        max_ts = self._passes
        if not round_idx:
            self._pol_mask_list = torch.cat([self._pol_mask_list for i in range(4)], dim=1)

        fw_idx, fw_weights = get_interpolation(
            self._event_list, self._flow_list, max_ts, self.res, self.flow_scaling, round_idx=round_idx
        )
        fw_iwe_pos = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=self._pol_mask_list[:, :, 0:1])
        fw_iwe_neg = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=self._pol_mask_list[:, :, 1:2])

        return torch.cat([fw_iwe_pos, fw_iwe_neg], dim=1)


class FWL(BaseValidationLoss):
    """
    Flow Warp Loss (FWL), as described in Section 4.1 of the paper 'Reducing the Sim-to-Real Gap for Event
    Cameras', Stoffregen and Scheerlinck et al., ECCV'20.
    The FWL metric is the ratio of the contrast of the image of warped events and that of the image of
    (non-warped) events; hence, the larger the value of this metric, the better the optical flow estimate.
    Contrast is measured through the spatial variance of the image of events.
    Note that this metric is sensitive to the number of input events.
    """

    def __init__(self, config, device, flow_scaling=128):
        super().__init__(config, device, flow_scaling)

    def forward(self):
        max_ts = self._passes

        # image of (forward) warped events
        fw_idx, fw_weights = get_interpolation(
            self._event_list, self._flow_list, max_ts, self.res, self.flow_scaling, round_idx=True
        )
        fw_IWE = interpolate(fw_idx.long(), fw_weights, self.res)

        # image of non-warped events
        zero_idx, zero_weights = get_interpolation(
            self._event_list, self._flow_list * 0, max_ts, self.res, self.flow_scaling, round_idx=True
        )
        IE = interpolate(zero_idx.long(), zero_weights, self.res)

        # Forward Warping Loss (FWL)
        FWL = spatial_variance(fw_IWE) / spatial_variance(IE)
        FWL = FWL.view((fw_IWE.shape[0]))  # one metric per batch

        return FWL


class RSAT(BaseValidationLoss):
    """
    Similarly to the FWL metric, the Ratio of the Squared Averaged Timestamps (RSAT) metric is the ratio of the squared sum
    of the per-pixel and per-polarity average timestamp of the image of warped events and that of
    the image of (non-warped) events; hence, the lower the value of this metric, the better the optical flow estimate.
    Note that this metric is sensitive to the number of input events.
    """

    def __init__(self, config, device, flow_scaling=128):
        super().__init__(config, device, flow_scaling)

    def forward(self):
        max_ts = self._passes

        # image of (forward) warped averaged timestamps
        ts_list = self._event_list[:, :, 0:1]
        fw_idx, fw_weights = get_interpolation(
            self._event_list, self._flow_list, max_ts, self.res, self.flow_scaling, round_idx=True
        )
        fw_iwe_pos = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=self._pol_mask_list[:, :, 0:1])
        fw_iwe_neg = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=self._pol_mask_list[:, :, 1:2])
        fw_iwe_pos_ts = interpolate(
            fw_idx.long(), fw_weights * ts_list, self.res, polarity_mask=self._pol_mask_list[:, :, 0:1]
        )
        fw_iwe_neg_ts = interpolate(
            fw_idx.long(), fw_weights * ts_list, self.res, polarity_mask=self._pol_mask_list[:, :, 1:2]
        )
        fw_iwe_pos_ts /= fw_iwe_pos + 1e-9
        fw_iwe_neg_ts /= fw_iwe_neg + 1e-9
        fw_iwe_pos_ts = fw_iwe_pos_ts / max_ts
        fw_iwe_neg_ts = fw_iwe_neg_ts / max_ts

        # image of non-warped averaged timestamps
        zero_idx, zero_weights = get_interpolation(
            self._event_list, self._flow_list * 0, max_ts, self.res, self.flow_scaling, round_idx=True
        )
        zero_iwe_pos = interpolate(
            zero_idx.long(), zero_weights, self.res, polarity_mask=self._pol_mask_list[:, :, 0:1]
        )
        zero_iwe_neg = interpolate(
            zero_idx.long(), zero_weights, self.res, polarity_mask=self._pol_mask_list[:, :, 1:2]
        )
        zero_iwe_pos_ts = interpolate(
            zero_idx.long(), zero_weights * ts_list, self.res, polarity_mask=self._pol_mask_list[:, :, 0:1]
        )
        zero_iwe_neg_ts = interpolate(
            zero_idx.long(), zero_weights * ts_list, self.res, polarity_mask=self._pol_mask_list[:, :, 1:2]
        )
        zero_iwe_pos_ts /= zero_iwe_pos + 1e-9
        zero_iwe_neg_ts /= zero_iwe_neg + 1e-9
        zero_iwe_pos_ts = zero_iwe_pos_ts / max_ts
        zero_iwe_neg_ts = zero_iwe_neg_ts / max_ts

        # (scaled) sum of the squares of the per-pixel and per-polarity average timestamps
        fw_iwe_pos_ts = fw_iwe_pos_ts.view(fw_iwe_pos_ts.shape[0], -1)
        fw_iwe_neg_ts = fw_iwe_neg_ts.view(fw_iwe_neg_ts.shape[0], -1)
        fw_iwe_pos_ts = torch.sum(fw_iwe_pos_ts ** 2, dim=1)
        fw_iwe_neg_ts = torch.sum(fw_iwe_neg_ts ** 2, dim=1)
        fw_ts_sum = fw_iwe_pos_ts + fw_iwe_neg_ts

        fw_nonzero_px = fw_iwe_pos + fw_iwe_neg
        fw_nonzero_px[fw_nonzero_px > 0] = 1
        fw_nonzero_px = fw_nonzero_px.view(fw_nonzero_px.shape[0], -1)
        fw_ts_sum /= torch.sum(fw_nonzero_px, dim=1)

        zero_iwe_pos_ts = zero_iwe_pos_ts.view(zero_iwe_pos_ts.shape[0], -1)
        zero_iwe_neg_ts = zero_iwe_neg_ts.view(zero_iwe_neg_ts.shape[0], -1)
        zero_iwe_pos_ts = torch.sum(zero_iwe_pos_ts ** 2, dim=1)
        zero_iwe_neg_ts = torch.sum(zero_iwe_neg_ts ** 2, dim=1)
        zero_ts_sum = zero_iwe_pos_ts + zero_iwe_neg_ts

        zero_nonzero_px = zero_iwe_pos + zero_iwe_neg
        zero_nonzero_px[zero_nonzero_px > 0] = 1
        zero_nonzero_px = zero_nonzero_px.view(zero_nonzero_px.shape[0], -1)
        zero_ts_sum /= torch.sum(zero_nonzero_px, dim=1)

        return fw_ts_sum / zero_ts_sum


class AEE(BaseValidationLoss):
    """
    Average endpoint error (which is just the Euclidean distance) loss.
    """

    def __init__(self, config, device, flow_scaling=128):
        super().__init__(config, device, flow_scaling)

    @property
    def num_events(self):
        return float("inf")

    def forward(self):

        # convert flow
        flow = self._flow_map[-1] * self.flow_scaling
        flow *= self._dt_gt.to(self.device) / self._dt_input.to(self.device)
        flow_mag = flow.pow(2).sum(1).sqrt()

        # compute AEE
        error = (flow - self._gtflow).pow(2).sum(1).sqrt()

        # AEE not computed in pixels without events
        event_mask = self._event_mask[:, -1, :, :].bool()

        # AEE not computed in pixels without valid ground truth
        gtflow_mask_x = self._gtflow[:, 0, :, :] == 0.0
        gtflow_mask_y = self._gtflow[:, 1, :, :] == 0.0
        gtflow_mask = gtflow_mask_x * gtflow_mask_y
        gtflow_mask = ~gtflow_mask

        # mask AEE and flow
        mask = event_mask * gtflow_mask
        mask = mask.view(self._flow_map[-1].shape[0], -1)
        error = error.view(self._flow_map[-1].shape[0], -1)
        flow_mag = flow_mag.view(self._flow_map[-1].shape[0], -1)
        error = error * mask
        flow_mag = flow_mag * mask

        # compute AEE and percentage of outliers
        num_valid_px = torch.sum(mask, dim=1)
        AEE = torch.sum(error, dim=1) / (num_valid_px + 1e-9)

        outliers = (error > 3.0) * (error > 0.05 * flow_mag)  # AEE larger than 3px and 5% of the flow magnitude
        percent_AEE = outliers.sum() / (num_valid_px + 1e-9)

        return AEE, percent_AEE
