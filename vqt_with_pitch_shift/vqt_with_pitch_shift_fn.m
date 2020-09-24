function [c_AC_ensemble, err_db] = vqt_with_pitch_shift_fn(varargin)
%% modification histroy
%{
* 20-Nov-2019: split long file into several parts
%}

    %{
    db_scale: true (default) or false
    mono: true (default) or false
    sub_sampling_factor: int, default to 22
    gamma: double, default to 14.112 Hz (equivalent to 9000 samples at sr of 44100) 
    max_shift_bins: int, default to 2
    wav_file: string
    gpu_idx: int, when gpu_idx >=1, do fft on gpu. Be advised that in Matlab, gpu index starts from 1. 
             default to -1, not use gpu.
    %}

    %% miscellaneous
    % read variable input arguments, if not present, set to default value
    
    partition_threshold = 300;  % seconds
    
    nvarargin = length(varargin);
    assert(mod(nvarargin, 2) == 0);
   
    for vid=1:2:nvarargin
        key = varargin{vid};
        value = varargin{vid + 1};
        assert(ischar(key));
        switch key
            case 'db_scale'
                db_scale = value;
            case 'mono'
                mono = value;
            case 'sub_sampling_factor'
                sub_sampling_factor = value;
            case 'gamma'
                gamma = value;
            case 'max_shift_bins'
                max_shift_bins = value;
            case 'wav_file'
                wav_file = value;
            case 'gpu_idx'
                gpu_idx = value;
            otherwise
                error('unknown key %s', key);
        end
    end
    
    if exist('db_scale', 'var') == 0
        db_scale = true;
    end
    
    if exist('mono', 'var') == 0
        mono = true;
    end
    
    if exist('sub_sampling_factor', 'var') == 0
        sub_sampling_factor = 22;
    end
    
    if exist('gamma', 'var') == 0
        gamma = 14.112;
    end
    
    if exist('max_shift_bins', 'var') == 0
        max_shift_bins = 2;
    end
    
    assert(exist('wav_file', 'var') == 1);
    
    if exist('gpu_idx', 'var') == 0
        gpu_idx = -1;
    end
    
    %check validity of parameters
    assert(isscalar(db_scale) && islogical(db_scale));
    assert(isscalar(mono) && islogical(mono));
    
    assert(isscalar(sub_sampling_factor));
    assert(isa(sub_sampling_factor, 'int64') || isa(sub_sampling_factor, 'double'));
    if isa(sub_sampling_factor, 'int64')
        sub_sampling_factor = double(sub_sampling_factor);
    end
    assert(mod(sub_sampling_factor, 1) == 0);
    
    assert(isscalar(gamma));
    assert(isa(gamma, 'int64') || isa(gamma, 'double'));
    if isa(gamma, 'int64')
       gamma = double(gamma); 
    end
    
    assert(isscalar(max_shift_bins));
    assert(isa(max_shift_bins, 'int64') || isa(max_shift_bins, 'double'));
    if isa(max_shift_bins, 'int64')
        max_shift_bins = double(max_shift_bins);
    end
    assert(max_shift_bins == 1 || max_shift_bins == 2 || max_shift_bins == 3);
    
    assert(isstring(wav_file) || ischar(wav_file));
    
    assert(isscalar(gpu_idx));
    assert(isa(gpu_idx, 'int64') || isa(gpu_idx, 'double'));
    if isa(gpu_idx, 'int64')
        gpu_idx = double(gpu_idx);
    end
    assert(mod(gpu_idx, 1) == 0);
    
    % check number of outputs
    assert(nargout == 1 || nargout == 2);
    if nargout == 2
        fprintf('\nsetting:\n');
        fprintf('dB scale - %d\n', db_scale);
        fprintf('mono - %d\n', mono);
        fprintf('sub-sampling factor - %d\n', sub_sampling_factor);
        fprintf('maximum shift bins - %d\n', max_shift_bins);
        fprintf('gamma - %.2f\n', gamma); 
        fprintf('gpu_idx - %d\n\n', gpu_idx);
    end
    
    sr = 44100;
    hop_size_hd = 128;
    hop_size_ld = 64;
    
    assert(mod(hop_size_hd, hop_size_ld) == 0);
    
    wav_info = audioinfo(wav_file);
    assert(wav_info.SampleRate == sr || wav_info.SampleRate == 48000);
    assert(wav_info.BitsPerSample == 16);
    
    x = audioread(wav_file, 'native');
    assert(ismatrix(x));
    assert(isa(x, 'int16'));
    x = double(x) / 32768;
    if mono && size(x, 2) >= 2
       x = mean(x, 2); 
    end
    
    if wav_info.SampleRate > sr
       x = resample(x, sr, wav_info.SampleRate);
       fprintf('warning: the original sample rate is %d and downsampled to %d\n', wav_info.SampleRate, sr);
    end
    
    num_chs = size(x, 2); 
    if mono
        assert(num_chs == 1); 
    end
    
    %% original
    num_samples = length(x);
    num_hops = ceil(num_samples / hop_size_ld);
    num_hops_down_sampled = ceil(num_hops / sub_sampling_factor);
    num_struct_original = struct(...
        'num_samples', num_samples, ...
        'num_hops', num_hops, ...
        'num_hops_down_sampled', num_hops_down_sampled);
    
    %% partition
    num_partitions = ceil(wav_info.Duration / partition_threshold);
    if num_partitions > 1
        fprintf('partition threshold - %.0f s\n', partition_threshold);
        fprintf('duration - %.0f s\n', wav_info.Duration);
        fprintf('num of partitions - %d\n', num_partitions);
    end
    
    num_hops_per_partition_before_padding_for_circular_shift = ceil(ceil(num_struct_original.num_samples / num_partitions) / hop_size_ld);
    if mod(num_hops_per_partition_before_padding_for_circular_shift, 2) == 1
        num_hops_per_partition_before_padding_for_circular_shift = num_hops_per_partition_before_padding_for_circular_shift + 1;
    end
    num_hops_after_partition = num_hops_per_partition_before_padding_for_circular_shift * num_partitions;
    assert(num_hops_after_partition >= num_struct_original.num_hops);
    num_hops_down_sampled_after_partition = ceil(num_hops_after_partition / sub_sampling_factor);
    assert(num_hops_down_sampled_after_partition >= num_struct_original.num_hops_down_sampled);
    num_samples_after_partition = num_hops_after_partition * hop_size_ld;
    num_padded_samples_to_enable_partition = num_samples_after_partition - num_struct_original.num_samples;
    if num_padded_samples_to_enable_partition > 0
       x = padarray(x, num_padded_samples_to_enable_partition, 'post');
       assert(isequal(size(x), [num_samples_after_partition, num_chs]));
    end
    num_struct_after_partition = struct(...
        'num_samples', num_samples_after_partition, ...
        'num_padded_samples', num_padded_samples_to_enable_partition, ...
        'num_partitions', num_partitions, ...
        'num_hops', num_hops_after_partition, ...
        'num_hops_per_partition_without_padding', num_hops_per_partition_before_padding_for_circular_shift, ...
        'num_hops_down_sampled', num_hops_down_sampled_after_partition);
    
    %% per partition
    corr_len = 2.88 * sr / gamma;
    num_padded_hops_per_partition_at_either_side = ceil(corr_len / 2 / hop_size_ld);
    if mod(num_padded_hops_per_partition_at_either_side, 2) == 1
        num_padded_hops_per_partition_at_either_side = num_padded_hops_per_partition_at_either_side + 1;
    end
    num_hops_per_partition_after_padding_for_circular_shift = num_struct_after_partition.num_hops_per_partition_without_padding + ...
        2 * num_padded_hops_per_partition_at_either_side;
    num_samples_per_partition_before_padding_for_circular_shift = num_struct_after_partition.num_hops_per_partition_without_padding * hop_size_ld;
    num_samples_per_partition_after_padding_for_circular_shift = num_hops_per_partition_after_padding_for_circular_shift * hop_size_ld;
    num_struct_per_partition = struct(...
        'num_samples_without_padding', num_samples_per_partition_before_padding_for_circular_shift, ...
        'num_samples_with_padding', num_samples_per_partition_after_padding_for_circular_shift, ...
        'padded_samples_at_either_side', num_padded_hops_per_partition_at_either_side * hop_size_ld, ...
        'padded_hops_at_either_side', num_padded_hops_per_partition_at_either_side, ...
        'num_hops_without_padding', num_samples_per_partition_before_padding_for_circular_shift / hop_size_ld, ...
        'num_hops_with_padding', num_samples_per_partition_after_padding_for_circular_shift / hop_size_ld);
    % the padding in each partition is to counteract the influence of circular shift
    Ls = num_struct_per_partition.num_samples_with_padding;
    
    %% high definition
    B_hd = 84;
    assert(mod(B_hd, 12) == 0);
    bins_per_note_hd = B_hd / 12;
    assert(mod(bins_per_note_hd, 2) == 1);
    fmin_hd = midi2freq(21) * 2 ^ (-(bins_per_note_hd - 1) / 2 / B_hd);
    fmax_hd = 1.8e4;
    [fbas_hd, cqtbw_hd] = hd_fbas_cqtbw_fn(fmin_hd, fmax_hd, gamma, B_hd, sr);
    [g_hd, fbas_hd] = nsgcqwin_with_dc_and_nyquist_fn(fbas_hd, cqtbw_hd, sr, Ls);
    assert(mod(Ls, hop_size_hd) == 0);
    M_AC_hd = Ls / hop_size_hd;
    assert(M_AC_hd >= max(cellfun(@length, g_hd(2:end - 1))));
    M_DC_hd = ceil(length(g_hd{1}) / hop_size_hd) * hop_size_hd;
    M_nyq_hd = ceil(length(g_hd{end}) / hop_size_hd) * hop_size_hd;
    Ms_hd = zeros(size(fbas_hd));
    Ms_hd(2:end-1) = M_AC_hd;
    Ms_hd(1) = M_DC_hd;
    Ms_hd(end) = M_nyq_hd;
    for idx=1:length(g_hd)
        g_hd{idx} = g_hd{idx} * (2 * Ms_hd(idx) / Ls);
    end

    gd_hd = nsdual_xian(g_hd, fbas_hd, Ms_hd, Ls);
    par_struct_hd = struct(...
        'c_DC', 0, ...
        'c_AC', 0, ...
        'c_nyq', 0, ...
        'g', {g_hd}, ...
        'gd', {gd_hd}, ...
        'M_DC', M_DC_hd, ...
        'M_AC', M_AC_hd, ...
        'M_nyq', M_nyq_hd, ...
        'fbas', fbas_hd ...   
    );

    %% low definition
    B_ld = 36;
    assert(mod(B_ld, 12) == 0);
    bins_per_note_ld = B_ld / 12;
    assert(mod(bins_per_note_ld, 2) == 1);
    fmin_ld = midi2freq(21) * 2 ^ (-(bins_per_note_ld - 1) / 2 / B_ld);
    fmax_ld = midi2freq(108 + 24) * 2 ^ ((bins_per_note_ld - 1) / 2 / B_ld);
    [fbas_ld, cqtbw_ld] = hd_fbas_cqtbw_fn(fmin_ld, fmax_ld, gamma, B_ld, sr);
    [g_ld, fbas_ld] = nsgcqwin_with_dc_and_nyquist_fn(fbas_ld, cqtbw_ld, sr, Ls);
    assert(length(g_ld) == (88 + 24) * bins_per_note_ld + 2);
    assert(mod(Ls, hop_size_ld) == 0);
    M_AC_ld = Ls / hop_size_ld;
    assert(M_AC_ld >= max(cellfun(@length, g_ld(2:end - 1))));
    M_DC_ld = ceil(length(g_ld{1}) / hop_size_ld) * hop_size_ld;
    M_nyq_ld = ceil(length(g_ld{end}) / hop_size_ld) * hop_size_ld;
    Ms_ld = zeros(size(fbas_ld));
    Ms_ld(1) = M_DC_ld;
    Ms_ld(2:end - 1) = M_AC_ld;
    Ms_ld(end) = M_nyq_ld;
    for idx =1:length(g_ld)
       g_ld{idx} = g_ld{idx} * (2 * Ms_ld(idx) / Ls); 
    end

    gd_ld = nsdual_xian(g_ld, fbas_ld, Ms_ld, Ls);
    par_struct_ld = struct(...
        'g', {g_ld}, ...
        'gd', {gd_ld}, ...
        'M_DC', M_DC_ld, ...
        'M_AC', M_AC_ld, ...
        'M_nyq', M_nyq_ld, ...
        'fbas', fbas_ld ...
    );

    if nargout == 2
        x_after_partition_recovered = zeros(size(x));
    end
    
    list_of_shift_bins = -max_shift_bins:max_shift_bins;
    num_freq_bins_ld = length(par_struct_ld.fbas) - 2;
    num_freq_bins_hd = length(par_struct_hd.fbas) - 2;
    num_shift_bins = length(list_of_shift_bins);
    c_AC_ensemble = zeros([num_struct_after_partition.num_hops_down_sampled, num_freq_bins_ld, num_chs, num_shift_bins], 'single');
    
    num_paddings_to_counteract_circular_shift = num_struct_per_partition.padded_samples_at_either_side;
    Ls_without_padding = num_struct_per_partition.num_samples_without_padding;
    for partition_idx = 1:num_partitions
        fprintf('\npartition - %d/%d\n', partition_idx, num_partitions);
        if partition_idx == 1
            partition_start_sample = 1;
            partition_end_sample = Ls_without_padding;
            x_partition = x(partition_start_sample:partition_end_sample, :);
            x_partition = padarray(x_partition, num_paddings_to_counteract_circular_shift, 'pre');
            if num_partitions > 1
                post_padding_start_sample = partition_end_sample + 1;
                post_padding_end_sample = partition_end_sample + num_paddings_to_counteract_circular_shift;
                assert(post_padding_end_sample < num_struct_after_partition.num_samples);
                post_padding_block = x(post_padding_start_sample:post_padding_end_sample, :);
                x_partition = [x_partition;post_padding_block];
            else
                assert(partition_end_sample == num_struct_after_partition.num_samples);
                x_partition = padarray(x_partition, num_paddings_to_counteract_circular_shift, 'post');
            end
        elseif partition_idx < num_partitions
            partition_start_sample = (partition_idx - 1) * Ls_without_padding + 1;
            partition_start_sample = partition_start_sample - num_paddings_to_counteract_circular_shift;
            assert(partition_start_sample > 0);
            partition_end_sample = partition_idx * Ls_without_padding;
            partition_end_sample = partition_end_sample + num_paddings_to_counteract_circular_shift;
            assert(partition_end_sample < num_struct_after_partition.num_samples);
            x_partition = x(partition_start_sample:partition_end_sample, :);
        else % partition_idx == num_partitions
            partition_start_sample = (partition_idx - 1) * Ls_without_padding + 1;
            partition_start_sample = partition_start_sample - num_paddings_to_counteract_circular_shift;
            assert(partition_start_sample > 0);
            partition_end_sample = partition_idx * Ls_without_padding;
            assert(partition_end_sample == num_struct_after_partition.num_samples);
            x_partition = x(partition_start_sample:partition_end_sample, :);
            x_partition = padarray(x_partition, num_paddings_to_counteract_circular_shift, 'post');
        end
        assert(isequal(size(x_partition),[Ls, num_chs]));
        
        % do pitch shift
        if gpu_idx <= 0
            x_partition = fft(x_partition, [], 1);
        else
            try
                gpuDevice(gpu_idx);
                x_partition = gather(fft(gpuArray(x_partition), [], 1));
            catch ME
                gpuDevice([]);
                rethrow(ME);
            end
        end
        disp('fft of x done');
        
        [c_DC, c_AC, c_nyq] = nsgtf_real_xian(...
            x_partition, ...
            par_struct_hd.g, ...
            par_struct_hd.fbas, ...
            par_struct_hd.M_DC, ...
            par_struct_hd.M_AC, ...
            par_struct_hd.M_nyq, ...
            'global', ...
            gpu_idx);
        assert(isequal(size(c_DC), [par_struct_hd.M_DC, num_chs]));
        if mono
            assert(isequal(size(c_AC), [par_struct_hd.M_AC, num_freq_bins_hd]));
        else
            assert(isequal(size(c_AC), [par_struct_hd.M_AC, num_freq_bins_hd, num_chs]));
        end
        assert(isequal(size(c_nyq), [par_struct_hd.M_nyq, num_chs]));
        
        par_struct_hd.c_DC = c_DC;
        par_struct_hd.c_AC = c_AC;
        par_struct_hd.c_nyq = c_nyq;
        disp('expansion for hd done');
        
        if partition_idx == 1
            c_AC_ensemble_start_idx = 1;
            offset_idx_for_c_AC_partition = 0;
        end
        
        num_hops_down_sampled_for_this_partition = ceil((num_struct_per_partition.num_hops_without_padding - offset_idx_for_c_AC_partition) / sub_sampling_factor);
        c_AC_ensemble_end_idx = c_AC_ensemble_start_idx + num_hops_down_sampled_for_this_partition - 1;
        if partition_idx == num_partitions
            assert(c_AC_ensemble_end_idx == num_struct_after_partition.num_hops_down_sampled);
        end
        c_AC_partition_start_idx = num_struct_per_partition.padded_hops_at_either_side + 1 + offset_idx_for_c_AC_partition;
        c_AC_partition_end_idx = c_AC_partition_start_idx + (num_hops_down_sampled_for_this_partition - 1) * sub_sampling_factor;
        assert(c_AC_partition_end_idx <= num_struct_per_partition.padded_hops_at_either_side + num_struct_per_partition.num_hops_without_padding);
        offset_idx_for_c_AC_partition = num_struct_per_partition.padded_hops_at_either_side + num_struct_per_partition.num_hops_without_padding - ...
            c_AC_partition_end_idx;
        assert(offset_idx_for_c_AC_partition >= 0 && offset_idx_for_c_AC_partition <= sub_sampling_factor - 1);
        offset_idx_for_c_AC_partition = sub_sampling_factor - 1 - offset_idx_for_c_AC_partition;
        
        num_hops_hd = Ls / hop_size_hd;
        num_hops_ld = Ls / hop_size_ld;
        for shift_idx = 1:length(list_of_shift_bins)
            shift_bins = list_of_shift_bins(shift_idx);
            if shift_bins == 0
                [c_DC, c_AC, c_nyq] = nsgtf_real_xian(...
                    x_partition, par_struct_ld.g, par_struct_ld.fbas, ...
                    par_struct_ld.M_DC, par_struct_ld.M_AC, ...
                    par_struct_ld.M_nyq, 'global', gpu_idx);
                
                if nargout == 2
                    y_partition = nsigtf_real_xian(c_DC, c_AC, c_nyq, par_struct_ld.gd, par_struct_ld.fbas, ...
                        Ls, 'global', gpu_idx);
                    assert(isequal([Ls, num_chs], size(y_partition)));
                    x_recovered_start_sample = (partition_idx - 1) * Ls_without_padding + 1;
                    x_recovered_end_sample = partition_idx * Ls_without_padding;
                    x_after_partition_recovered(x_recovered_start_sample:x_recovered_end_sample, :) = ...
                        y_partition(num_struct_per_partition.padded_samples_at_either_side + 1: ...
                        end - num_struct_per_partition.padded_samples_at_either_side, :);
                end
         
            else % shift_bins != 0
                c_AC = pitch_shift_xian(par_struct_hd.c_AC, par_struct_hd.fbas(2:end - 1), shift_bins, Ls, 1e-6);
                if mono
                    assert(isequal(size(c_AC),[num_hops_hd, num_freq_bins_hd]));
                else
                    assert(isequal(size(c_AC),[num_hops_hd, num_freq_bins_hd, num_chs]));
                end
                
                x_partition_shifted = reconstruct_fft_from_coeffs_and_gd_fn(...
                    par_struct_hd.c_DC, c_AC, par_struct_hd.c_nyq, ...
                    par_struct_hd.gd, par_struct_hd.fbas, Ls, 'global', gpu_idx);
                assert(length(x_partition_shifted) == Ls);
                [~, c_AC] = nsgtf_real_xian(...
                    x_partition_shifted, par_struct_ld.g, par_struct_ld.fbas, par_struct_ld.M_DC, par_struct_ld.M_AC, ...
                    par_struct_ld.M_nyq, 'global', gpu_idx);
            end
            
            if mono
                assert(isequal(size(c_AC), [num_hops_ld, num_freq_bins_ld]));
            else
                assert(isequal(size(c_AC), [num_hops_ld, num_freq_bins_ld, num_chs]));
            end
            
            c_AC = c_AC(c_AC_partition_start_idx:sub_sampling_factor:c_AC_partition_end_idx, :, :);
            if db_scale
                c_AC = single(20 * log10(abs(c_AC) + 1e-10) + 200);
            else
                c_AC = single(abs(c_AC));
            end
            assert(isa(c_AC, 'single'));
            c_AC_ensemble(c_AC_ensemble_start_idx:c_AC_ensemble_end_idx, :, :, shift_idx) = c_AC;
            fprintf('shift bins - %d done\n', shift_bins);
            
        end % end shift_idx
        
        c_AC_ensemble_start_idx = c_AC_ensemble_end_idx + 1;
        
        if gpu_idx >= 1
            gpuDevice([]); 
        end
        
        
    end % partition_idx
    
    if num_struct_after_partition.num_hops_down_sampled > num_struct_original.num_hops_down_sampled
        c_AC_ensemble = c_AC_ensemble(1:num_struct_original.num_hops_down_sampled, :, :, :);  
    end
    
    
    if nargout == 2
        x = x(1:num_struct_original.num_samples, :);
        x_after_partition_recovered = x_after_partition_recovered(1:num_struct_original.num_samples, :);
        x = x(:);
        x_after_partition_recovered = x_after_partition_recovered(:);
        err_db = 20 * log10(norm(x(:)) / norm(x(:) - x_after_partition_recovered(:)));
    end
end