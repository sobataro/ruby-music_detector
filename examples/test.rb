require 'music_detector'

if ARGV.count == 0
  puts("usage: ruby test.rb [test files...]")
  exit
end

config = MusicDetector::Configuration.new
fv_extractor = MusicDetector::FeatureVectorExtractor.new(config)

# import pre-trained model
regression = MusicDetector::MultipleLinearRegression.import_from('./model')

@positive_count = 0
@negative_count = 0

def test(file:, fv_extractor:, regression:)
  feature_vector = fv_extractor.extract_from(file: file, seektime: 0, duration: 3.2)
#  p feature_vector
  estimation = regression.estimate(feature_vector).first
  puts("#{estimation}: #{file}")

  if estimation
    @positive_count += 1
  else
    @negative_count += 1
  end
end

ARGV.each do |file|
  if FileTest.directory?(file)
    Dir.entries(file)
      .map { |f| "#{file}/#{f}" }
      .select { |f| f.end_with?('.wav') }
      .each { |f| test(file: f, fv_extractor: fv_extractor, regression: regression) }
  else
    test(file: file, fv_extractor: fv_extractor, regression: regression)
  end
end

puts("positive: #{@positive_count}, negative: #{@negative_count}")
