SELECT(year/10) * 10 AS decade,
original_language,
COUNT(*)AS movie_count
FROM movies
GROUP BY decade, original_language
ORDER BY decade ASC, movie_count DESC;