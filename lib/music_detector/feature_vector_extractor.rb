require 'wavefile'
require 'numru/fftw3'

module MusicDetector
  class FeatureVectorExtractor

    attr_reader :config

    # @param [MusicDetector::Configuration] config config to process music file
    def initialize(config)
      @config = config
    end

    # @param [String] file      path of the input audio file
    # @param [Float]  seektime  seek time in the audio file (in seconds)
    # @param [Float]  duration  duration in the audio file (in seconds)
    # @return [NArray<Float>]   extracted feature vector
    def extract_from(file:, seektime:, duration:)
      # read audio file
      wave, samplerate = read_audio_file(file: file, seektime: seektime, duration: duration)

      # do fft and make frequency spectrum
      length      = wave.length / 2
      spectrum    = NumRu::FFTW3.fft(wave, NumRu::FFTW3::FORWARD).abs[0...length]
      frequencies = NArray.to_na((0...length).map { |i| i.to_f / length * samplerate / 2 }) # frequencies of each element in spectrum

      # bandpath filter (to faster computation)
      spectrum, frequencies = band_path_filter(spectrum: spectrum, frequencies: frequencies)

      # prepare for analysis
      log_frequencies = NMath::log(frequencies)
      log_temperament = NMath::log(equal_temperament)

      log_bin_freq_half_bandwidth    = (log_temperament[1] - log_temperament[0]) / 2.0
      log_in_tune_freq_threshold     = log_bin_freq_half_bandwidth * @config.in_tune_cents / 100.0
      log_out_of_tune_freq_threshold = log_bin_freq_half_bandwidth * @config.out_of_tune_cents / 100.0

      log_temperament.map do |log_bin_center_freq|
        # indices of the target bin (for spectrum)
        log_bin_indices = (log_frequencies - log_bin_center_freq).abs < log_bin_freq_half_bandwidth

        # extract the target bin
        target_log_frequencies = log_frequencies[log_bin_indices]
        target_spectrum        = spectrum[log_bin_indices]

        # calc ratio between in-tune and out-of-tune powers
        target_in_tune_indices     = (target_log_frequencies - log_bin_center_freq).abs <= log_in_tune_freq_threshold
        target_out_of_tune_indices = (target_log_frequencies - log_bin_center_freq).abs >  log_out_of_tune_freq_threshold

        in_tune_power     = target_spectrum[target_in_tune_indices].mean
        out_of_tune_power = target_spectrum[target_out_of_tune_indices].mean

        in_tune_power / out_of_tune_power
      end.sort.reverse
    end

    private

    def read_audio_file(file:, seektime:, duration:)
      mono_wave = nil
      samplerate = nil

      WaveFile::Reader.new(file) do |sound|
        samplerate = sound.format.sample_rate
        channels = sound.format.channels

        # seek
        sound.read((seektime * samplerate).round)

        # read
        sample_count = (duration * samplerate).round
        mono_wave = NArray.sint(sample_count)

        sound.read(sample_count).samples.each.with_index do |sample, i|
          case sample
          when Array
            mono_wave[i] = sample.inject(&:+) / channels # normalize to monoral
            mono_wave[i] *= (2 ** (bits_per_sample - 1)) if Float === sample.first
          when Fixnum
            mono_wave[i] = sample
          when Float
            mono_wave[i] = sample * (2 ** (bits_per_sample - 1))
          else
            raise StandardError("unsupported file: #{file}")
          end
        end
      end

      [mono_wave, samplerate]
    end

    def band_path_filter(spectrum:, frequencies:)
      hpf_freq = @config.a * 2 ** ((@config.temperament_range.first - 1) / 12.0)
      lpf_freq = @config.a * 2 ** ((@config.temperament_range.last + 1) / 12.0)

      bpf_indices = (hpf_freq < frequencies) * (frequencies < lpf_freq)
      spectrum    = spectrum[bpf_indices]
      frequencies = frequencies[bpf_indices]
      [spectrum, frequencies]
    end

    def equal_temperament
      NArray.to_na(@config.temperament_range.map { |i| @config.a * 2 ** (i / 12.0) })
    end
  end
end
