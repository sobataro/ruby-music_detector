require 'narray'

module MusicDetector

  # y = X * b + err
  class MultipleLinearRegression
    NEGATIVE = -1
    POSITIVE = 1

    attr_accessor :b

    # @param [NArray<Float>] negative_example_fvs  feature vectors of negative examples
    # @param [NArray<Float>] positive_example_fvs  feature vectors of positive examples
    # @return [MultipleLinearRegression]           model trained by given examples
    # FIXME: change parameters to:
    #  - a matrix of example fvs
    #  - a vector of labels of the matrix
    def self.train_by(negative_example_fvs:, positive_example_fvs:)
      model = MultipleLinearRegression.new

      negative_count = negative_example_fvs.count
      positive_count = positive_example_fvs.count
      raise RuntimeError('give one or more negative examples') if negative_count == 0
      raise RuntimeError('give one or more positive examples') if positive_count == 0

      fv_length = negative_example_fvs.first.total
      negative_example_fvs.each { |fv| raise RuntimeError if fv_length != fv.total }
      positive_example_fvs.each { |fv| raise RuntimeError if fv_length != fv.total }

      # prepare matrix X and vector y
      x = NMatrix.float(fv_length + 1, negative_count + positive_count)
      y = NVector.float(negative_count + positive_count)

      x[0, true] = NVector.int(negative_count + positive_count).fill(1)
      negative_example_fvs.each.with_index do |fv, i|
        x[1..fv_length, i] = fv
        y[i] = NEGATIVE
      end
      positive_example_fvs.each.with_index do |fv, i|
        x[1..fv_length, negative_count + i] = fv
        y[negative_count + i] = POSITIVE
      end

      # estimate parameters
      model.b = (x.transpose * x).inverse * x.transpose * y

      model
    end

    def self.import_from(file_path)
      model = MultipleLinearRegression.new

      lines = File.read(file_path).each_line.to_a

      # skip configurations
      lines.shift

      # vector b
      b = lines.first.chomp.split(',').map { |e| e.to_f }
      model.b = NVector[*b]

      model
    end

    # @param [NArray]
    def estimate(input)
      raise RuntimeError('the form of the input is differ from trained model') if @b.total - 1 != input.shape[0]

      x = NMatrix.float(input.shape[0] + 1, input.shape[1])
      (input.shape[1]).times do |i|
        x[0, i] = 1.0
        x[1..(input.shape[0]), i] = input[true, i].flatten
      end
      x * @b
    end

    def export_to(file_path, config:)
      File.open(file_path, 'w+') do |file|
        file.puts(config.inspect)
        file.puts(b.to_a.join(','))
      end
    end

    private

    def initialize
    end
  end
end
