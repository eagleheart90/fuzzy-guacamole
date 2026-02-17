SELECT title, year,lb_rating,profit
FROM movies
WHERE original_language = 'ja'
AND lb_rating > 3.8
AND profit > 0
ORDER by lb_rating DESC;