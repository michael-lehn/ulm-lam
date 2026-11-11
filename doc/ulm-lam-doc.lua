-- ulm-lam-doc.lua: helpers for ulm-lam.tex (LuaLaTeX).
--
-- Excerpts are cut out of the real sources at typesetting time, so the
-- document cannot drift away from the ODS files and examples.

local M = {}

local function slurp_lines(path)
  local f, err = io.open(path, "r")
  if not f then tex.error("ulmlamdoc: cannot open " .. path .. ": " .. err) end
  local lines = {}
  for line in f:lines() do lines[#lines + 1] = line end
  f:close()
  return lines
end

local function write(outname, lines)
  local out = assert(io.open(outname, "w"))
  out:write(table.concat(lines, "\n"), "\n")
  out:close()
end

-- The TableGen record `def <name>`: up to the closing "}" in column 0 if
-- the record has a body, otherwise up to the line ending in ";".
function M.def(path, name, outname)
  local lines, got, brace = slurp_lines(path), {}, false
  for _, line in ipairs(lines) do
    if #got == 0 then
      if line:match("^def%s+" .. name .. "[%s:<{;]")
          or line:match("^def%s+" .. name .. "$") then
        got[1] = line
        brace = line:find("{", 1, true) ~= nil
        if not brace and line:match(";%s*$") then break end
      end
    else
      got[#got + 1] = line
      brace = brace or line:find("{", 1, true) ~= nil
      if brace and line:match("^}") then break end
      if not brace and line:match(";%s*$") then break end
    end
  end
  if #got == 0 then
    tex.error("ulmlamdoc: def " .. name .. " not found in " .. path)
  end
  write(outname, got)
end

-- Lines from the first line containing `from` (plain text) up to, but not
-- including, the next line containing `to` (to the end if `to` is empty).
function M.between(path, from, to, outname)
  local lines, got, on = slurp_lines(path), {}, false
  for _, line in ipairs(lines) do
    if on and to ~= "" and line:find(to, 1, true) then break end
    if not on and line:find(from, 1, true) then on = true end
    if on then got[#got + 1] = line end
  end
  if #got == 0 then
    tex.error("ulmlamdoc: '" .. from .. "' not found in " .. path)
  end
  -- drop trailing blank and separator lines
  while #got > 0 and (got[#got]:match("^%s*$") or got[#got]:match("^//%-%-")) do
    got[#got] = nil
  end
  write(outname, got)
end

-- Inline code: typeset verbatim (catcode 12) and allow line breaks after
-- "_", ".", "-" and "/" so long names and paths do not overflow.
function M.code(s)
  s = s:gsub("\\([#{}_%%])", "%1"):gsub("##", "#")
  for ch in s:gmatch(utf8.charpattern) do
    tex.sprint(-2, ch)
    if ch == "_" or ch == "." or ch == "-" or ch == "/" then
      tex.sprint("\\allowbreak{}")
    end
  end
end

return M
