require 'music_detector'

if ARGV.count != 3
  puts("usage: ruby build_model.rb [n for n-fold cross validation] [dir of negative examples] [dir of positive examples]")
  exit
end

k = ARGV[0].to_i
negative_example_dir = ARGV[1]
positive_example_dir = ARGV[2]
negative_example_dir = negative_example_dir + '/' unless negative_example_dir.end_with?('/')
positive_example_dir = positive_example_dir + '/' unless positive_example_dir.end_with?('/')

config = MusicDetector::Configuration.new

fv_extractor = MusicDetector::FeatureVectorExtractor.new(config)
negative_fvs = Dir.entries(negative_example_dir)
                 .map { |f| "#{negative_example_dir}#{f}" }
                 .select { |f| f.end_with?('.wav') }
                 .map { |f| p f; fv_extractor.extract_from(file: f, seektime: 0, duration: 3.2) }
positive_fvs = Dir.entries(positive_example_dir)
                 .map { |f| "#{positive_example_dir}#{f}" }
                 .select { |f| f.end_with?('.wav') }
                 .map { |f| p f; fv_extractor.extract_from(file: f, seektime: 0, duration: 3.2) }

fv_length = negative_fvs.first.total

# do k-fold cross validation
(0...k).each do |i|
  # feature vectors for training
  train_negative, train_positive, test_negative, test_positive = [[], [], [], []]
  negative_fvs.each.with_index do |nfv, j|
    if j % k == i
      train_negative << nfv
    else
      test_negative << nfv
    end
  end
  positive_fvs.each.with_index do |nfv, j|
    if j % k == i
      train_positive << nfv
    else
      test_positive << nfv
    end
  end

  # build model
  regression = MusicDetector::MultipleLinearRegression.train_by(negative_example_fvs: train_negative, positive_example_fvs: train_positive)

  # test the built model
  x = NMatrix.float(fv_length, test_negative.count + test_positive.count)
  y = NVector.float(test_negative.count + test_positive.count)

  test_negative.each.with_index do |fv, i|
    x[true, i] = fv
    y[i] = MusicDetector::MultipleLinearRegression::NEGATIVE
  end
  test_positive.each.with_index do |fv, i|
    x[true, test_negative.count + i] = fv
    y[test_negative.count + i] = MusicDetector::MultipleLinearRegression::POSITIVE
  end

  calculated_y = regression.estimate(x)

  tp, fp, tn, fn = [0, 0, 0, 0]
  calculated_y.each.with_index do |r, i|
    tp += 1 if r && y[i] == MusicDetector::MultipleLinearRegression::POSITIVE
    fp += 1 if r && y[i] == MusicDetector::MultipleLinearRegression::NEGATIVE
    fn += 1 if !r && y[i] == MusicDetector::MultipleLinearRegression::POSITIVE
    tn += 1 if !r && y[i] == MusicDetector::MultipleLinearRegression::NEGATIVE
  end
  puts("##{i} cross validation:")
  puts("tp=#{tp}, fp=#{fp}, fn=#{fn}, tn=#{tn}")
  puts("accuracy=#{(tp + tn).to_f / (tp + fp + fn + tn)}, precision=#{tp.to_f / (tp + fp)}, recall=#{tp.to_f / (tp + fn)}")
  puts('')
end
