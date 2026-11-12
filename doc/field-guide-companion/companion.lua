-- companion.lua: inline code for ulm-lam-companion.tex.
--
-- \code{...} gets its argument detokenized; this undoes what TeX needs in
-- the source (\% for %, \# for #, the doubled # of \detokenize) and allows
-- a line break after , . _ / ( and before ->, so that long MLIR types break.
local M = {}
local bs = string.char(92)

-- Print a piece of code; a long identifier may also break before an
-- upper-case letter that follows a lower-case one (CamelCase).
local function emit(piece)
  if #piece <= 18 then
    tex.sprint(-2, piece)
    return
  end
  local last = 1
  for k = 2, #piece do
    local a, b = piece:sub(k - 1, k - 1), piece:sub(k, k)
    if a:match("%l") and b:match("%u") and k - last >= 6 then
      tex.sprint(-2, piece:sub(last, k - 1))
      tex.sprint(bs .. "allowbreak{}")
      last = k
    end
  end
  tex.sprint(-2, piece:sub(last))
end

function M.code(s)
  s = s:gsub(bs .. "%%", "%%")
  s = s:gsub(bs .. "#", "#")
  s = s:gsub("##", "#")
  s = s:gsub(bs .. "&", "&")
  s = s:gsub(bs .. "_", "_")
  s = s:gsub(bs .. "{", "{")
  s = s:gsub(bs .. "}", "}")
  s = s:gsub(bs .. "textbackslash ?", bs)
  local i = 1
  local n = #s
  while i <= n do
    local j = s:find("[,%._/%(]", i)
    local k = s:find("%->", i)
    local cut
    if j and (not k or j < k) then
      emit(s:sub(i, j))
      cut = j + 1
    elseif k then
      emit(s:sub(i, k - 1))
      tex.sprint(bs .. "allowbreak{}")
      tex.sprint(-2, "->")
      cut = k + 2
    else
      emit(s:sub(i))
      break
    end
    if cut <= n then tex.sprint(bs .. "allowbreak{}") end
    i = cut
  end
end

return M
