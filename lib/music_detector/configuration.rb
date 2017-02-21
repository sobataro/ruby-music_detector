module MusicDetector
  class Configuration
    attr_reader :a, :temperament_range, :in_tune_cents, :out_of_tune_cents

    # @param [Numeric] a                  base A4 frequency of the equal temperament (typically 440)
    # @param [Range]   temperament_range  range of the equal temperament used to extract feature vector
    # @param [Float]   in_tune_cents      maximum frequency difference between an in-tune note and the equal temperament
    # @param [Float]   out_of_tune_ratio  minimum frequency difference between an out-of-tune note and the equal temperament
    def initialize(a: 440, temperament_range: -12..24, in_tune_cents: 10, out_of_tune_cents: 30)
      @a                 = a
      @temperament_range = temperament_range
      @in_tune_cents     = in_tune_cents
      @out_of_tune_cents = out_of_tune_cents
    end
  end
end
