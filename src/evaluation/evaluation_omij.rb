# frozen_string_literal: true

require 'json'
require 'uri'
require 'multiset'
require 'mustache'
require 'csv'

#################################################################
current_directory = __dir__

input_user = ARGV[0]
benchmark_name = input_user.split('.json').first

systemname = ARGV[2].nil? ? '' : ARGV[2]
config     = ARGV[3].nil? ? '' : ARGV[3]

template   = File.join(current_directory, 'evaluation_result_html.mustache')

#################################################################

def fscore(precision, recall)
  precision.zero? && recall.zero? ? 0 : (2 * precision * recall) / (precision + recall)
end

def normalize(answer)
  return "#{answer}.0" if answer =~ /\A\d+\z/

  URI.unescape(answer.strip)
end

def compare_answers(answers_user, answers_gold)
  results = {}

  answers_gold.each do |id, gold|
    results[id] = { total_gold: gold.length }
    # if id == 40
    #   puts "gold"
    #   puts gold
    # end
    if answers_user.key? id
      user = answers_user[id]
      # puts "user"
      # puts user
      results[id][:processed]  = true
      results[id][:total_user] = user.length
      results[id][:correct]    = (user.select { |x| gold.include? x }).length

    else
      results[id][:processed] = false
    end
  end

  results
end

def compute_results(results, answers_gold)
  breakdown = []

  sum_precision = 0
  sum_recall    = 0

  sum_correct         = 0
  sum_total_gold_all  = 0
  sum_total_gold_proc = 0
  sum_total_user      = 0

  results.each do |id, comp|
    sum_total_gold_all += comp[:total_gold]

    if comp[:processed]
      sum_total_gold_proc += comp[:total_gold]
      sum_total_user      += comp[:total_user]
      sum_correct         += comp[:correct]

      # OUT OF SCOPE questions
      if (comp[:total_gold]).zero?
        if (comp[:total_user]).zero?
          precision = 1
          recall    = 1
          f1        = 1
        else
          precision = 0
          recall    = 0
          f1        = 0
        end
      # all other questions
      else
        precision = if (comp[:total_user]).zero?
                      1
                    else
                      comp[:correct].to_f / comp[:total_user]
                    end
        recall = comp[:correct].to_f / comp[:total_gold]
        f1 = fscore(precision, recall)
      end
    else
      precision = 0
      recall    = 0
      f1        = 0
    end

    breakdown << { id: id,
                   precision: precision,
                   recall: recall,
                   f1: f1 }

    sum_precision += precision
    sum_recall    += recall
  end

  measures = { breakdown: breakdown.sort_by { |x| x[:id] }, processed: {}, all: {} }

  ## Measures on processed questions only

  number_of_processed_questions = (results.select { |_, comp| comp[:processed] }).length

  micro_precision1 = sum_total_user.zero? ? 0 : sum_correct.to_f / sum_total_user
  micro_recall1    = sum_total_gold_proc.zero? ? 0 : sum_correct.to_f / sum_total_gold_proc

  measures[:processed][:micro] = { precision: micro_precision1,
                                   recall: micro_recall1,
                                   f1: fscore(micro_precision1, micro_recall1) }

  macro_precision1 = number_of_processed_questions.zero? ? 0 : sum_precision.to_f / number_of_processed_questions
  macro_recall1    = number_of_processed_questions.zero? ? 0 : sum_recall.to_f / number_of_processed_questions

  measures[:processed][:macro] = { precision: macro_precision1,
                                   recall: macro_recall1,
                                   f1: fscore(macro_precision1, macro_recall1) }

  ## Measures on all questions

  # Micro

  total_number_of_questions = answers_gold.keys.length

  micro_precision2 = sum_total_user.zero? ? 0 : sum_correct.to_f / sum_total_user
  micro_recall2    = sum_total_gold_all.zero? ? 0 : sum_correct.to_f / sum_total_gold_all

  measures[:all][:micro] = { precision: micro_precision2,
                             recall: micro_recall2,
                             f1: fscore(micro_precision2, micro_recall2) }

  # Macro
  macro_precision2 = total_number_of_questions.zero? ? 0 : sum_precision.to_f / total_number_of_questions
  macro_recall2    = total_number_of_questions.zero? ? 0 : sum_recall.to_f / total_number_of_questions

  measures[:all][:macro] = { precision: macro_precision2,
                             recall: macro_recall2,
                             f1: fscore(macro_precision2, macro_recall2) }

  measures
end

def append_to_csv(benchmark_name, result, current_directory)
  precision = (result[:all][:macro][:precision] * 100).round(2)
  recall = (result[:all][:macro][:recall] * 100).round(2)
  f1 = (result[:all][:macro][:f1] * 100).round(2)

  csv_data = [
    [benchmark_name],
    %w[P R F1],
    [precision, recall, f1]
  ]

  csv_path = File.join(current_directory, 'data', 'evaluation_results.csv')
  CSV.open(csv_path, 'a+') do |csv|
    csv_data.each do |row|
      csv << row
    end
  end
end

def read_answers(file)
  out = {}
  gold_out = {}

  doc = JSON.parse(File.read(file, encoding: 'utf-8'))

  doc['questions'].each do |question|
    next if question.nil?

    answers = []
    if question.key?('answers') && question['answers']
      question['answers'].each do |answer|
        next if answer.nil?

        if answer.key?('results') && answer['results'].key?('bindings')
          answer['results']['bindings'].each do |bind|
            b = Multiset.new
            bind.each_value { |value| b << normalize(value['value']) }
            answers << b
          end
        elsif answer.key? 'boolean'
          answers << answer['boolean']
        end
      end
    end

    gold_answers = []
    if question.key?('golden_answers') && question['golden_answers']
      question['golden_answers'].each do |answer|
        next if answer.nil?
        if answer.key?('results') && answer['results'].key?('bindings')
          answer['results']['bindings'].each do |bind|
            b = Multiset.new
            bind.each_value { |value| b << normalize(value['value']) }
            gold_answers << b
          end
        elsif answer.key? 'boolean'
          gold_answers << answer['boolean']
        end
      end
    end

    id = question['id']
    id = id.to_i if id.is_a? String
    out[id] = answers.uniq
    gold_out[id] = gold_answers.uniq
  end

  [doc['dataset']['id'], out, gold_out]
end

#################################################################

dataset, answers_user, answers_gold = read_answers(input_user)
# _,       answers_gold = read_answers(File.join(current_directory, "data", dataset+".json"))

if answers_user.nil?
  html = { error: "Error when reading input file: #{input_user}" }
else

  results = compute_results(compare_answers(answers_user, answers_gold), answers_gold)

  html = Mustache.render(File.read(template), { gold: "#{dataset}.json",
                                                system: systemname,
                                                config: config,
                                                time: Time.now,
                                                results: results })

  ## append the results to comman csv for benchmark table generation
  append_to_csv(benchmark_name, results, current_directory)
end

file = "#{input_user}.html"
File.write(file, html)
puts file
